const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');
const { query } = require('../db');
const { authenticate } = require('../middleware/auth');
const {
  sendForgotPasswordEmail,
  sendEmailChangeConfirmEmail,
  sendPasswordChangeConfirmEmail,
  send2FACodeEmail,
} = require('../services/email');

const router = express.Router();

const FRONTEND_URL = process.env.FRONTEND_URL || 'https://beathole.com';

// Register
router.post('/register', async (req, res, next) => {
  try {
    const { email, password, username, displayName } = req.body;

    if (!email || !password || !username) {
      return res.status(400).json({ error: 'Email, password, and username are required' });
    }

    if (password.length < 8) {
      return res.status(400).json({ error: 'Password must be at least 8 characters' });
    }

    const existing = await query(
      'SELECT id FROM users WHERE email = ? OR username = ?',
      [email.toLowerCase(), username.toLowerCase()]
    );

    if (existing.rows.length > 0) {
      return res.status(409).json({ error: 'Email or username already taken' });
    }

    const passwordHash = await bcrypt.hash(password, 12);
    const id = uuidv4();

    await query(
      'INSERT INTO users (id, email, password_hash, username, display_name) VALUES (?, ?, ?, ?, ?)',
      [id, email.toLowerCase(), passwordHash, username.toLowerCase(), displayName || username]
    );

    const result = await query(
      'SELECT id, email, username, display_name, role, avatar_url, created_at FROM users WHERE id = ?',
      [id]
    );

    const user = result.rows[0];
    const token = jwt.sign({ userId: user.id }, process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRES_IN || '7d',
    });

    res.status(201).json({ user, token });
  } catch (err) {
    next(err);
  }
});

// Login
router.post('/login', async (req, res, next) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }

    const result = await query(
      'SELECT * FROM users WHERE email = ? AND is_active = 1',
      [email.toLowerCase()]
    );

    const user = result.rows[0];

    if (!user || !(await bcrypt.compare(password, user.password_hash))) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // 2FA check
    if (user.two_fa_enabled) {
      const code = String(Math.floor(100000 + Math.random() * 900000));
      const tokenId = uuidv4();
      const expiresAt = new Date(Date.now() + 10 * 60 * 1000);

      await query(
        `INSERT INTO auth_tokens (id, user_id, type, token, expires_at) VALUES (?, ?, '2fa_code', ?, ?)`,
        [tokenId, user.id, code, expiresAt]
      );

      try {
        await send2FACodeEmail({
          to: user.email,
          displayName: user.display_name || user.username,
          code,
        });
      } catch (e) {
        console.error('2FA email failed:', e.message);
      }

      return res.json({ requires2fa: true, userId: user.id });
    }

    const token = jwt.sign({ userId: user.id }, process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRES_IN || '7d',
    });

    const { password_hash, ...safeUser } = user;
    res.json({ user: safeUser, token });
  } catch (err) {
    next(err);
  }
});

// Get current user
router.get('/me', authenticate, async (req, res) => {
  res.json({ user: req.user });
});

// Update profile
router.put('/profile', authenticate, async (req, res, next) => {
  try {
    const { displayName, bio, avatarUrl } = req.body;

    await query(
      `UPDATE users SET
        display_name = COALESCE(?, display_name),
        bio = COALESCE(?, bio),
        avatar_url = COALESCE(?, avatar_url)
       WHERE id = ?`,
      [displayName || null, bio || null, avatarUrl || null, req.user.id]
    );

    const result = await query(
      'SELECT id, email, username, display_name, bio, avatar_url, role FROM users WHERE id = ?',
      [req.user.id]
    );

    res.json({ user: result.rows[0] });
  } catch (err) {
    next(err);
  }
});

// Public profile
router.get('/profile/:username', async (req, res, next) => {
  try {
    const result = await query(
      `SELECT u.id, u.username, u.display_name, u.avatar_url, u.bio, u.created_at,
              COUNT(DISTINCT b.id) as published_beats,
              COALESCE(SUM(b.play_count), 0) as total_plays
       FROM users u
       LEFT JOIN beats b ON b.creator_id = u.id AND b.status = 'published'
       WHERE u.username = ? AND u.is_active = 1
       GROUP BY u.id`,
      [req.params.username.toLowerCase()]
    );
    if (!result.rows[0]) return res.status(404).json({ error: 'User not found' });
    const user = result.rows[0];

    const beatsResult = await query(
      `SELECT id, title, genre, mood, bpm, cover_art_url, play_count, status, created_at
       FROM beats WHERE creator_id = ? AND status = 'published'
       ORDER BY created_at DESC LIMIT 20`,
      [user.id]
    );

    res.json({ user, beats: beatsResult.rows });
  } catch (err) { next(err); }
});

// Change email (legacy direct update — kept for backwards compatibility)
router.put('/change-email', authenticate, async (req, res, next) => {
  try {
    const { newEmail, currentPassword } = req.body;
    if (!newEmail || !currentPassword) return res.status(400).json({ error: 'New email and current password are required' });

    const result = await query('SELECT password_hash FROM users WHERE id = ?', [req.user.id]);
    if (!await bcrypt.compare(currentPassword, result.rows[0].password_hash)) {
      return res.status(401).json({ error: 'Incorrect password' });
    }

    const existing = await query('SELECT id FROM users WHERE email = ? AND id != ?', [newEmail.toLowerCase(), req.user.id]);
    if (existing.rows.length > 0) return res.status(409).json({ error: 'Email already in use' });

    await query('UPDATE users SET email = ? WHERE id = ?', [newEmail.toLowerCase(), req.user.id]);
    res.json({ message: 'Email updated' });
  } catch (err) { next(err); }
});

// Change username / display name
router.put('/change-username', authenticate, async (req, res, next) => {
  try {
    const { username, displayName } = req.body;
    if (!username) return res.status(400).json({ error: 'Username is required' });
    if (username.length < 3 || username.length > 30) return res.status(400).json({ error: 'Username must be 3–30 characters' });
    if (!/^[a-zA-Z0-9_]+$/.test(username)) return res.status(400).json({ error: 'Username can only contain letters, numbers, and underscores' });

    const existing = await query('SELECT id FROM users WHERE username = ? AND id != ?', [username.toLowerCase(), req.user.id]);
    if (existing.rows.length > 0) return res.status(409).json({ error: 'Username already taken' });

    await query(
      'UPDATE users SET username = ?, display_name = COALESCE(?, display_name) WHERE id = ?',
      [username.toLowerCase(), displayName || null, req.user.id]
    );

    const updated = await query('SELECT id, email, username, display_name, bio, avatar_url, role FROM users WHERE id = ?', [req.user.id]);
    res.json({ user: updated.rows[0] });
  } catch (err) { next(err); }
});

// Change password (legacy direct update — kept for backwards compatibility)
router.put('/change-password', authenticate, async (req, res, next) => {
  try {
    const { currentPassword, newPassword, confirmPassword } = req.body;
    if (!currentPassword || !newPassword) return res.status(400).json({ error: 'Current and new password are required' });
    if (newPassword !== confirmPassword) return res.status(400).json({ error: 'Passwords do not match' });
    if (newPassword.length < 8) return res.status(400).json({ error: 'Password must be at least 8 characters' });

    const result = await query('SELECT password_hash FROM users WHERE id = ?', [req.user.id]);
    if (!await bcrypt.compare(currentPassword, result.rows[0].password_hash)) {
      return res.status(401).json({ error: 'Current password is incorrect' });
    }

    const newHash = await bcrypt.hash(newPassword, 12);
    await query('UPDATE users SET password_hash = ? WHERE id = ?', [newHash, req.user.id]);
    res.json({ message: 'Password updated successfully' });
  } catch (err) { next(err); }
});

// Forgot password
router.post('/forgot-password', async (req, res, next) => {
  try {
    const { emailOrUsername } = req.body;
    if (!emailOrUsername) return res.status(400).json({ error: 'Email or username is required' });

    const result = await query(
      'SELECT id, email, display_name, username FROM users WHERE email = ? OR username = ? AND is_active = 1',
      [emailOrUsername.toLowerCase(), emailOrUsername.toLowerCase()]
    );

    // Always return success to avoid user enumeration
    if (result.rows.length > 0) {
      const user = result.rows[0];
      const token = crypto.randomBytes(32).toString('hex');
      const tokenId = uuidv4();
      const expiresAt = new Date(Date.now() + 60 * 60 * 1000);

      await query(
        `INSERT INTO auth_tokens (id, user_id, type, token, expires_at) VALUES (?, ?, 'forgot_password', ?, ?)`,
        [tokenId, user.id, token, expiresAt]
      );

      const resetLink = `${FRONTEND_URL}/reset-password/${token}`;
      try {
        await sendForgotPasswordEmail({
          to: user.email,
          displayName: user.display_name || user.username,
          resetLink,
        });
      } catch (e) {
        console.error('Forgot password email failed:', e.message);
      }
    }

    res.json({ message: "If an account with this email/username exists, you'll receive a reset link." });
  } catch (err) { next(err); }
});

// Reset password
router.post('/reset-password', async (req, res, next) => {
  try {
    const { token, newPassword } = req.body;
    if (!token || !newPassword) return res.status(400).json({ error: 'Token and new password are required' });
    if (newPassword.length < 8) return res.status(400).json({ error: 'Password must be at least 8 characters' });

    const result = await query(
      `SELECT * FROM auth_tokens WHERE token = ? AND type = 'forgot_password' AND used = 0 AND expires_at > NOW()`,
      [token]
    );

    if (result.rows.length === 0) {
      return res.status(400).json({ error: 'Invalid or expired reset token' });
    }

    const authToken = result.rows[0];
    const newHash = await bcrypt.hash(newPassword, 12);

    await query('UPDATE users SET password_hash = ? WHERE id = ?', [newHash, authToken.user_id]);
    await query('UPDATE auth_tokens SET used = 1 WHERE id = ?', [authToken.id]);

    res.json({ message: 'Password reset successfully' });
  } catch (err) { next(err); }
});

// Request email change (sends confirmation to current email)
router.post('/request-email-change', authenticate, async (req, res, next) => {
  try {
    const { newEmail, currentPassword } = req.body;
    if (!newEmail || !currentPassword) return res.status(400).json({ error: 'New email and current password are required' });

    const userResult = await query('SELECT password_hash, email, display_name, username FROM users WHERE id = ?', [req.user.id]);
    const user = userResult.rows[0];

    if (!await bcrypt.compare(currentPassword, user.password_hash)) {
      return res.status(401).json({ error: 'Incorrect password' });
    }

    const existing = await query('SELECT id FROM users WHERE email = ? AND id != ?', [newEmail.toLowerCase(), req.user.id]);
    if (existing.rows.length > 0) return res.status(409).json({ error: 'Email already in use' });

    const token = crypto.randomBytes(32).toString('hex');
    const tokenId = uuidv4();
    const expiresAt = new Date(Date.now() + 60 * 60 * 1000);

    await query(
      `INSERT INTO auth_tokens (id, user_id, type, token, data, expires_at) VALUES (?, ?, 'verify_email_change', ?, ?, ?)`,
      [tokenId, req.user.id, token, JSON.stringify({ newEmail: newEmail.toLowerCase() }), expiresAt]
    );

    const confirmLink = `${FRONTEND_URL}/confirm-email-change/${token}`;
    try {
      await sendEmailChangeConfirmEmail({
        to: user.email,
        displayName: user.display_name || user.username,
        newEmail: newEmail.toLowerCase(),
        confirmLink,
      });
    } catch (e) {
      console.error('Email change confirm email failed:', e.message);
    }

    res.json({ message: 'A confirmation link has been sent to your current email address.' });
  } catch (err) { next(err); }
});

// Confirm email change via token
router.post('/confirm-email-change/:token', async (req, res, next) => {
  try {
    const { token } = req.params;

    const result = await query(
      `SELECT * FROM auth_tokens WHERE token = ? AND type = 'verify_email_change' AND used = 0 AND expires_at > NOW()`,
      [token]
    );

    if (result.rows.length === 0) {
      return res.status(400).json({ error: 'Invalid or expired token' });
    }

    const authToken = result.rows[0];
    const data = typeof authToken.data === 'string' ? JSON.parse(authToken.data) : authToken.data;

    await query('UPDATE users SET email = ? WHERE id = ?', [data.newEmail, authToken.user_id]);
    await query('UPDATE auth_tokens SET used = 1 WHERE id = ?', [authToken.id]);

    res.json({ message: 'Email updated successfully' });
  } catch (err) { next(err); }
});

// Request password change (sends confirmation email)
router.post('/request-password-change', authenticate, async (req, res, next) => {
  try {
    const { currentPassword, newPassword } = req.body;
    if (!currentPassword || !newPassword) return res.status(400).json({ error: 'Current and new password are required' });
    if (newPassword.length < 8) return res.status(400).json({ error: 'Password must be at least 8 characters' });

    const userResult = await query('SELECT password_hash, email, display_name, username FROM users WHERE id = ?', [req.user.id]);
    const user = userResult.rows[0];

    if (!await bcrypt.compare(currentPassword, user.password_hash)) {
      return res.status(401).json({ error: 'Current password is incorrect' });
    }

    const newPasswordHash = await bcrypt.hash(newPassword, 12);
    const token = crypto.randomBytes(32).toString('hex');
    const tokenId = uuidv4();
    const expiresAt = new Date(Date.now() + 60 * 60 * 1000);

    await query(
      `INSERT INTO auth_tokens (id, user_id, type, token, data, expires_at) VALUES (?, ?, 'verify_password_change', ?, ?, ?)`,
      [tokenId, req.user.id, token, JSON.stringify({ newPasswordHash }), expiresAt]
    );

    const confirmLink = `${FRONTEND_URL}/confirm-password-change/${token}`;
    try {
      await sendPasswordChangeConfirmEmail({
        to: user.email,
        displayName: user.display_name || user.username,
        confirmLink,
      });
    } catch (e) {
      console.error('Password change confirm email failed:', e.message);
    }

    res.json({ message: 'A confirmation link has been sent to your email address.' });
  } catch (err) { next(err); }
});

// Confirm password change via token
router.post('/confirm-password-change/:token', async (req, res, next) => {
  try {
    const { token } = req.params;

    const result = await query(
      `SELECT * FROM auth_tokens WHERE token = ? AND type = 'verify_password_change' AND used = 0 AND expires_at > NOW()`,
      [token]
    );

    if (result.rows.length === 0) {
      return res.status(400).json({ error: 'Invalid or expired token' });
    }

    const authToken = result.rows[0];
    const data = typeof authToken.data === 'string' ? JSON.parse(authToken.data) : authToken.data;

    await query('UPDATE users SET password_hash = ? WHERE id = ?', [data.newPasswordHash, authToken.user_id]);
    await query('UPDATE auth_tokens SET used = 1 WHERE id = ?', [authToken.id]);

    res.json({ message: 'Password updated successfully' });
  } catch (err) { next(err); }
});

// Toggle 2FA
router.post('/2fa/toggle', authenticate, async (req, res, next) => {
  try {
    const { enabled } = req.body;
    if (typeof enabled !== 'boolean') return res.status(400).json({ error: 'enabled must be a boolean' });

    await query('UPDATE users SET two_fa_enabled = ? WHERE id = ?', [enabled ? 1 : 0, req.user.id]);
    res.json({ message: `Two-factor authentication ${enabled ? 'enabled' : 'disabled'}`, two_fa_enabled: enabled });
  } catch (err) { next(err); }
});

// Verify 2FA code
router.post('/2fa/verify', async (req, res, next) => {
  try {
    const { userId, code } = req.body;
    if (!userId || !code) return res.status(400).json({ error: 'userId and code are required' });

    const result = await query(
      `SELECT * FROM auth_tokens WHERE token = ? AND user_id = ? AND type = '2fa_code' AND used = 0 AND expires_at > NOW()`,
      [String(code), userId]
    );

    if (result.rows.length === 0) {
      return res.status(401).json({ error: 'Invalid or expired code' });
    }

    const authToken = result.rows[0];
    await query('UPDATE auth_tokens SET used = 1 WHERE id = ?', [authToken.id]);

    const userResult = await query(
      'SELECT id, email, username, display_name, role, avatar_url, bio, two_fa_enabled, created_at FROM users WHERE id = ? AND is_active = 1',
      [userId]
    );

    if (userResult.rows.length === 0) {
      return res.status(401).json({ error: 'User not found' });
    }

    const user = userResult.rows[0];
    const token = jwt.sign({ userId: user.id }, process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRES_IN || '7d',
    });

    res.json({ user, token });
  } catch (err) { next(err); }
});

module.exports = router;
