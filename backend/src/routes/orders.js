const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { query } = require('../db');
const { authenticate } = require('../middleware/auth');
const stripeService = require('../services/stripe');
const { sendBeatSoldEmail, sendBuyerReceiptEmail } = require('../services/email');

const router = express.Router();

/**
 * Maak een checkout sessie voor het kopen van een licentie
 */
router.post('/checkout', authenticate, async (req, res, next) => {
  try {
    const { beatId, licenseId } = req.body;

    if (!beatId || !licenseId) {
      return res.status(400).json({ error: 'beatId and licenseId are required' });
    }

    const beatResult = await query(
      `SELECT b.*, u.username as creator_username, u.stripe_account_id as creator_stripe_account
       FROM beats b
       JOIN users u ON b.creator_id = u.id
       WHERE b.id = ? AND b.status = 'published'`,
      [beatId]
    );

    if (!beatResult.rows[0]) {
      return res.status(404).json({ error: 'Beat not found' });
    }

    const beat = beatResult.rows[0];

    const licenseResult = await query(
      'SELECT * FROM licenses WHERE id = ? AND beat_id = ? AND is_active = 1',
      [licenseId, beatId]
    );

    if (!licenseResult.rows[0]) {
      return res.status(404).json({ error: 'License not found' });
    }

    const license = licenseResult.rows[0];

    const feeResult = await query(
      "SELECT `value` FROM platform_settings WHERE `key` = 'platform_fee_percent'"
    );
    const platformFeePercent = parseInt(feeResult.rows[0]?.value || '20');

    const platformFeeCents = Math.round(license.price_cents * (platformFeePercent / 100));
    const creatorEarningsCents = license.price_cents - platformFeeCents;

    const orderId = uuidv4();
    const downloadToken = uuidv4();

    await query(
      `INSERT INTO orders (id, buyer_id, beat_id, license_id, amount_cents, platform_fee_cents, creator_earnings_cents, status, download_token)
       VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)`,
      [orderId, req.user.id, beatId, licenseId, license.price_cents, platformFeeCents, creatorEarningsCents, downloadToken]
    );

    const orderResult = await query('SELECT * FROM orders WHERE id = ?', [orderId]);
    const order = orderResult.rows[0];

    const session = await stripeService.createCheckoutSession({
      beat,
      license,
      buyer: req.user,
      order,
    });

    await query(
      'UPDATE orders SET stripe_session_id = ? WHERE id = ?',
      [session.id, orderId]
    );

    res.json({ sessionId: session.id, url: session.url });
  } catch (err) {
    next(err);
  }
});

/**
 * Stripe webhook handler — betalingen verwerken
 */
router.post('/webhook', express.raw({ type: 'application/json' }), async (req, res) => {
  const sig = req.headers['stripe-signature'];

  let event;
  try {
    event = stripeService.constructWebhookEvent(req.body, sig);
  } catch (err) {
    console.error('Webhook signature verification failed:', err.message);
    return res.status(400).json({ error: `Webhook error: ${err.message}` });
  }

  if (event.type === 'checkout.session.completed') {
    const session = event.data.object;
    if (session.mode === 'payment') {
      await handleSuccessfulPayment(session);
    }
  }

  res.json({ received: true });
});

const handleSuccessfulPayment = async (session) => {
  try {
    const { order_id, beat_id, license_id } = session.metadata;

    // Update order: betaald + download window 48 uur
    await query(
      `UPDATE orders SET
        status = 'completed',
        stripe_payment_intent_id = ?,
        download_expires_at = DATE_ADD(NOW(), INTERVAL 48 HOUR)
       WHERE id = ?`,
      [session.payment_intent, order_id]
    );

    // Haal verdiensten op
    const orderResult = await query(
      'SELECT creator_earnings_cents FROM orders WHERE id = ?',
      [order_id]
    );

    const { creator_earnings_cents } = orderResult.rows[0];

    // Voeg verdiensten toe aan creator balans
    const beatResult = await query(
      'SELECT creator_id FROM beats WHERE id = ?',
      [beat_id]
    );

    if (beatResult.rows[0]) {
      const creatorId = beatResult.rows[0].creator_id;
      await query(
        `UPDATE users SET
          balance_cents = balance_cents + ?,
          total_earnings_cents = total_earnings_cents + ?
         WHERE id = ?`,
        [creator_earnings_cents, creator_earnings_cents, creatorId]
      );

      // Bij exclusive licentie: beat van marketplace afhalen
      const licenseResult = await query(
        'SELECT type, name, features FROM licenses WHERE id = ?',
        [license_id]
      );

      if (licenseResult.rows[0]?.type === 'exclusive') {
        await query(
          "UPDATE beats SET status = 'removed' WHERE id = ?",
          [beat_id]
        );
        await query(
          'UPDATE licenses SET is_active = 0 WHERE beat_id = ?',
          [beat_id]
        );
      }

      // Send emails to creator and buyer
      try {
        const beatInfoRes = await query(
          `SELECT b.title, u.email as creator_email, u.display_name as creator_display_name, u.username as creator_username
           FROM beats b JOIN users u ON b.creator_id = u.id WHERE b.id = ?`,
          [beat_id]
        );
        const orderInfoRes = await query(
          `SELECT o.amount_cents, u.email as buyer_email, u.display_name as buyer_display_name, u.username as buyer_username
           FROM orders o JOIN users u ON o.buyer_id = u.id WHERE o.id = ?`,
          [order_id]
        );
        const bi = beatInfoRes.rows[0];
        const oi = orderInfoRes.rows[0];
        const li = licenseResult.rows[0];
        if (bi && oi) {
          await sendBeatSoldEmail({
            to: bi.creator_email,
            creatorName: bi.creator_display_name || bi.creator_username,
            beatTitle: bi.title,
            licenseType: li?.name || li?.type || 'License',
            earningsCents: creator_earnings_cents,
            buyerUsername: oi.buyer_username,
          });
          await sendBuyerReceiptEmail({
            to: oi.buyer_email,
            buyerName: oi.buyer_display_name || oi.buyer_username,
            beatTitle: bi.title,
            licenseType: li?.name || li?.type || 'License',
            licenseFeatures: li?.features,
            amountCents: oi.amount_cents,
            creatorUsername: bi.creator_username,
            beatId: beat_id,
          });
        }
      } catch (e) { console.error('Order emails failed (non-fatal):', e.message); }
    }

    console.log(`✅ Betaling voltooid voor order ${order_id} — €${(creator_earnings_cents / 100).toFixed(2)} naar creator`);
  } catch (err) {
    console.error('Error handling payment:', err);
  }
};

/**
 * Order success details
 */
router.get('/success', authenticate, async (req, res, next) => {
  try {
    const { session_id } = req.query;

    const orderResult = await query(
      `SELECT o.*, b.title as beat_title, b.mp3_url, b.wav_url,
              l.name as license_name, l.type as license_type, l.features
       FROM orders o
       JOIN beats b ON o.beat_id = b.id
       JOIN licenses l ON o.license_id = l.id
       WHERE o.stripe_session_id = ? AND o.buyer_id = ?`,
      [session_id, req.user.id]
    );

    if (!orderResult.rows[0]) {
      return res.status(404).json({ error: 'Order not found' });
    }

    res.json(orderResult.rows[0]);
  } catch (err) {
    next(err);
  }
});

/**
 * Aankoopgeschiedenis van de koper
 */
router.get('/my-purchases', authenticate, async (req, res, next) => {
  try {
    const result = await query(
      `SELECT o.*, b.title as beat_title, b.mp3_url, b.wav_url, b.preview_url,
              l.name as license_name, l.type as license_type,
              u.username as creator_username
       FROM orders o
       JOIN beats b ON o.beat_id = b.id
       JOIN licenses l ON o.license_id = l.id
       JOIN users u ON b.creator_id = u.id
       WHERE o.buyer_id = ? AND o.status = 'completed'
       ORDER BY o.created_at DESC`,
      [req.user.id]
    );

    res.json(result.rows);
  } catch (err) {
    next(err);
  }
});

const CREDIT_PACKAGES = [
  { id: 'credits_50',   credits: 50,   price_cents: 499,  label: '50 Beat Credits' },
  { id: 'credits_250',  credits: 250,  price_cents: 1499, label: '250 Beat Credits' },
  { id: 'credits_1000', credits: 1000, price_cents: 5999, label: '1000 Beat Credits' },
];

// GET /orders/credit-packages
router.get('/credit-packages', (req, res) => {
  res.json(CREDIT_PACKAGES);
});

// POST /orders/buy-credits — Stripe checkout for beat credit bundles
router.post('/buy-credits', authenticate, async (req, res, next) => {
  try {
    const { packageId } = req.body;
    const pkg = CREDIT_PACKAGES.find(p => p.id === packageId);
    if (!pkg) return res.status(400).json({ error: 'Invalid package' });

    const stripe = require('../services/stripe');
    const session = await stripe.createCreditCheckoutSession({
      userId: req.user.id,
      packageId: pkg.id,
      credits: pkg.credits,
      priceCents: pkg.price_cents,
      label: pkg.label,
      successUrl: `${process.env.FRONTEND_URL}/credits/success?session_id={CHECKOUT_SESSION_ID}`,
      cancelUrl: `${process.env.FRONTEND_URL}/credits`,
    });

    res.json({ url: session.url });
  } catch (err) {
    next(err);
  }
});

// GET /orders/credits-success — confirm credit purchase and add to account
router.get('/credits-success', authenticate, async (req, res, next) => {
  try {
    const { session_id } = req.query;
    if (!session_id) return res.status(400).json({ error: 'session_id required' });

    const stripeService = require('../services/stripe');
    const session = await stripeService.getSession(session_id);
    if (session.payment_status !== 'paid') {
      return res.status(400).json({ error: 'Payment not completed' });
    }

    // Metadata contains credits and userId
    const { credits, userId, packageId } = session.metadata || {};
    if (!credits || userId !== req.user.id) {
      return res.status(400).json({ error: 'Invalid session' });
    }

    // Prevent double-credit by checking if session already processed
    const existing = await query('SELECT id FROM orders WHERE stripe_session_id = ?', [session_id]);
    if (existing.rows.length > 0) {
      return res.json({ message: 'Already processed', credits: parseInt(credits) });
    }

    // Add credits
    await query('UPDATE users SET extra_beat_credits = extra_beat_credits + ? WHERE id = ?', [parseInt(credits), req.user.id]);

    // Record as order (for audit)
    const { v4: uuidv4 } = require('uuid');
    await query(
      `INSERT INTO orders (id, buyer_id, beat_id, license_id, stripe_session_id, amount_cents, platform_fee_cents, creator_earnings_cents, status)
       VALUES (?, ?, 'credits', 'credits', ?, ?, 0, 0, 'completed')`,
      [uuidv4(), req.user.id, session_id, session.amount_total || 0]
    );

    res.json({ message: 'Credits added', credits: parseInt(credits) });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
