const express = require('express');
const { query } = require('../db');
const { authenticate } = require('../middleware/auth');
const stripeService = require('../services/stripe');
const { sendProWelcomeEmail } = require('../services/email');

const router = express.Router();

/**
 * Current subscription status
 */
router.get('/status', authenticate, async (req, res, next) => {
  try {
    const result = await query(
      `SELECT subscription_plan, subscription_status, subscription_expires_at,
              beats_generated_count, beats_generated_reset_at, stripe_subscription_id
       FROM users WHERE id = ?`,
      [req.user.id]
    );

    const user = result.rows[0];
    if (!user) return res.status(404).json({ error: 'User not found' });

    const now = new Date();
    const isPro =
      user.subscription_plan === 'pro' &&
      user.subscription_status === 'active' &&
      (!user.subscription_expires_at || new Date(user.subscription_expires_at) > now);

    const limitKey = isPro ? 'pro_plan_beat_limit' : 'free_plan_beat_limit';
    const settingRes = await query('SELECT `value` FROM platform_settings WHERE `key` = ?', [limitKey]);
    const limit = parseInt(settingRes.rows[0]?.value || (isPro ? '500' : '10'));

    let currentCount = user.beats_generated_count;
    if (isPro) {
      const resetAt = user.beats_generated_reset_at ? new Date(user.beats_generated_reset_at) : new Date(0);
      const daysSinceReset = (now - resetAt) / (1000 * 60 * 60 * 24);
      if (daysSinceReset >= 30) {
        await query('UPDATE users SET beats_generated_count = 0, beats_generated_reset_at = NOW() WHERE id = ?', [req.user.id]);
        currentCount = 0;
      }
    }

    res.json({
      plan: user.subscription_plan,
      status: user.subscription_status,
      isPro,
      expiresAt: user.subscription_expires_at,
      beatsGenerated: currentCount,
      beatsLimit: limit,
      beatsRemaining: Math.max(0, limit - currentCount),
      resetAt: user.beats_generated_reset_at,
    });
  } catch (err) {
    next(err);
  }
});

/**
 * Start Pro subscription checkout
 */
router.post('/checkout', authenticate, async (req, res, next) => {
  try {
    const priceRes = await query(
      "SELECT `value` FROM platform_settings WHERE `key` = 'stripe_pro_price_id'"
    );
    const priceId = priceRes.rows[0]?.value || process.env.STRIPE_PRO_PRICE_ID;

    if (!priceId) {
      return res.status(500).json({ error: 'Subscription not configured. Contact support.' });
    }

    const session = await stripeService.createSubscriptionSession({
      user: req.user,
      priceId,
    });

    res.json({ sessionId: session.id, url: session.url });
  } catch (err) {
    next(err);
  }
});

/**
 * Stripe Customer Portal — manage subscription, payment method, invoices
 */
router.post('/portal', authenticate, async (req, res, next) => {
  try {
    const result = await query('SELECT stripe_customer_id FROM users WHERE id = ?', [req.user.id]);
    const customerId = result.rows[0]?.stripe_customer_id;

    if (!customerId) {
      return res.status(400).json({ error: 'No Stripe customer found. Purchase a subscription first.' });
    }

    const session = await stripeService.stripe.billingPortal.sessions.create({
      customer: customerId,
      return_url: `${process.env.FRONTEND_URL}/pricing`,
    });

    res.json({ url: session.url });
  } catch (err) {
    next(err);
  }
});

/**
 * Cancel subscription (at end of billing period)
 */
router.post('/cancel', authenticate, async (req, res, next) => {
  try {
    const result = await query(
      'SELECT stripe_subscription_id FROM users WHERE id = ?',
      [req.user.id]
    );

    const subId = result.rows[0]?.stripe_subscription_id;
    if (!subId) {
      return res.status(400).json({ error: 'No active subscription found' });
    }

    await stripeService.cancelSubscription(subId);

    await query(
      "UPDATE users SET subscription_status = 'canceled' WHERE id = ?",
      [req.user.id]
    );

    res.json({ message: 'Subscription will be canceled at the end of the billing period.' });
  } catch (err) {
    next(err);
  }
});

/**
 * Fallback: activeer subscription via session_id (als webhook faalt)
 */
router.post('/verify-session', authenticate, async (req, res, next) => {
  try {
    const { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: 'sessionId required' });

    const session = await stripeService.stripe.checkout.sessions.retrieve(sessionId);

    if (session.payment_status !== 'paid' || session.mode !== 'subscription') {
      return res.status(400).json({ error: 'Session not paid or not a subscription' });
    }

    const userId = session.metadata?.user_id;
    if (userId !== req.user.id) {
      return res.status(403).json({ error: 'Session does not belong to this user' });
    }

    const sub = await stripeService.retrieveSubscription(session.subscription);
    const expiresAt = new Date(sub.current_period_end * 1000);

    await query(
      `UPDATE users SET
        subscription_plan = 'pro',
        subscription_status = 'active',
        stripe_subscription_id = ?,
        stripe_customer_id = ?,
        subscription_expires_at = ?,
        beats_generated_count = 0,
        beats_generated_reset_at = NOW()
       WHERE id = ?`,
      [sub.id, session.customer, expiresAt, req.user.id]
    );

    res.json({ activated: true });
  } catch (err) {
    next(err);
  }
});

/**
 * Stripe webhook for subscription events
 */
router.post('/webhook', express.raw({ type: 'application/json' }), async (req, res) => {
  const sig = req.headers['stripe-signature'];

  let event;
  try {
    event = stripeService.constructWebhookEvent(
      req.body,
      sig,
      process.env.STRIPE_SUBSCRIPTION_WEBHOOK_SECRET || process.env.STRIPE_WEBHOOK_SECRET
    );
  } catch (err) {
    console.error('Subscription webhook error:', err.message);
    return res.status(400).json({ error: `Webhook error: ${err.message}` });
  }

  try {
    switch (event.type) {
      case 'checkout.session.completed': {
        const session = event.data.object;
        if (session.mode === 'subscription') {
          const userId = session.metadata?.user_id;
          const subscriptionId = session.subscription;

          if (userId && subscriptionId) {
            const sub = await stripeService.retrieveSubscription(subscriptionId);
            const expiresAt = new Date(sub.current_period_end * 1000);

            await query(
              `UPDATE users SET
                subscription_plan = 'pro',
                subscription_status = 'active',
                stripe_subscription_id = ?,
                stripe_customer_id = ?,
                subscription_expires_at = ?,
                beats_generated_count = 0,
                beats_generated_reset_at = NOW()
               WHERE id = ?`,
              [subscriptionId, session.customer, expiresAt, userId]
            );
            console.log(`✅ Pro subscription activated for user ${userId}`);
            // Send Pro welcome email
            try {
              const userRes = await query('SELECT email, display_name, username FROM users WHERE id = ?', [userId]);
              const u = userRes.rows[0];
              if (u) await sendProWelcomeEmail({ to: u.email, displayName: u.display_name || u.username });
            } catch (e) { console.error('Pro welcome email failed (non-fatal):', e.message); }
          }
        }
        break;
      }

      case 'customer.subscription.updated': {
        const sub = event.data.object;
        const expiresAt = new Date(sub.current_period_end * 1000);
        await query(
          `UPDATE users SET
            subscription_status = ?,
            subscription_expires_at = ?
           WHERE stripe_subscription_id = ?`,
          [sub.status, expiresAt, sub.id]
        );
        break;
      }

      case 'customer.subscription.deleted': {
        const sub = event.data.object;
        await query(
          `UPDATE users SET
            subscription_plan = 'free',
            subscription_status = 'active',
            stripe_subscription_id = NULL,
            subscription_expires_at = NULL
           WHERE stripe_subscription_id = ?`,
          [sub.id]
        );
        console.log(`Subscription ${sub.id} canceled — user downgraded to free`);
        break;
      }

      case 'invoice.payment_failed': {
        const invoice = event.data.object;
        await query(
          "UPDATE users SET subscription_status = 'past_due' WHERE stripe_customer_id = ?",
          [invoice.customer]
        );
        break;
      }
    }
  } catch (err) {
    console.error('Subscription webhook handler error:', err);
  }

  res.json({ received: true });
});

module.exports = router;
