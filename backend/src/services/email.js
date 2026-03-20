const nodemailer = require('nodemailer');

let transporter = null;

function getTransporter() {
  if (transporter) return transporter;
  if (!process.env.SMTP_HOST || !process.env.SMTP_USER || !process.env.SMTP_PASS) {
    console.warn('[Email] SMTP not configured — emails will be skipped');
    return null;
  }
  transporter = nodemailer.createTransport({
    host: process.env.SMTP_HOST,
    port: parseInt(process.env.SMTP_PORT || '587'),
    secure: process.env.SMTP_SECURE === 'true',
    auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
  });
  return transporter;
}

const FROM = process.env.EMAIL_FROM || 'BeatHole <noreply@beathole.com>';
const FRONTEND = process.env.FRONTEND_URL || 'https://beathole.com';

async function send(opts) {
  const t = getTransporter();
  if (!t) return;
  await t.sendMail({ from: FROM, ...opts });
}

// ─── Studio Invite Email ──────────────────────────────────────────────────────
async function sendStudioInviteEmail({ to, inviteeName, inviterName, beatTitle, beatId }) {
  const link = `${FRONTEND}/studio/invite/${beatId}`;
  await send({
    to,
    subject: `${inviterName} invited you to collaborate on "${beatTitle}"`,
    html: `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;background:#0f0f0f;color:#fff;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:28px;font-weight:900;letter-spacing:4px;color:#ff4444">BEATHOLE</span>
      </div>
      <h2 style="color:#fff;margin-bottom:8px">Studio Invitation 🎛️</h2>
      <p style="color:#aaa">Hey ${inviteeName},</p>
      <p style="color:#aaa"><strong style="color:#fff">${inviterName}</strong> has invited you to collaborate in the studio on:</p>
      <div style="background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:16px;margin:20px 0;text-align:center">
        <span style="font-size:20px;font-weight:700;color:#ff4444">🎵 ${beatTitle}</span>
      </div>
      <div style="text-align:center;margin:28px 0">
        <a href="${link}" style="background:#ff4444;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px">
          View Invitation
        </a>
      </div>
      <p style="color:#666;font-size:12px;text-align:center">BeatHole · AI Beat Generation Platform</p>
    </div>`,
  });
}

// ─── Pro Welcome Email ────────────────────────────────────────────────────────
async function sendProWelcomeEmail({ to, displayName }) {
  await send({
    to,
    subject: 'Welcome to BeatHole Pro 🎵',
    html: `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;background:#0f0f0f;color:#fff;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:28px;font-weight:900;letter-spacing:4px;color:#ff4444">BEATHOLE</span>
      </div>
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:48px">👑</span>
      </div>
      <h2 style="color:#ffd700;text-align:center;margin-bottom:8px">Welcome to Pro!</h2>
      <p style="color:#aaa;text-align:center">Hey ${displayName}, you've unlocked the full BeatHole experience.</p>
      <div style="background:#1a1a1a;border:1px solid #ffd70033;border-radius:8px;padding:20px;margin:24px 0">
        <h3 style="color:#ffd700;margin:0 0 12px">Your Pro benefits:</h3>
        <ul style="color:#aaa;margin:0;padding-left:20px;line-height:2">
          <li>🎵 500 beats per month</li>
          <li>🎛️ Full studio access with stems</li>
          <li>📤 Publish &amp; sell on the marketplace</li>
          <li>🤝 Studio collaboration features</li>
          <li>💰 80% revenue on every sale</li>
          <li>🔑 Download WAV + MP3</li>
        </ul>
      </div>
      <div style="text-align:center;margin:28px 0">
        <a href="${FRONTEND}/generate" style="background:#ff4444;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px">
          Start Generating
        </a>
      </div>
      <p style="color:#666;font-size:12px;text-align:center">BeatHole · AI Beat Generation Platform</p>
    </div>`,
  });
}

// ─── Beat Sold Email (to creator) ────────────────────────────────────────────
async function sendBeatSoldEmail({ to, creatorName, beatTitle, licenseType, earningsCents, buyerUsername }) {
  const earnings = (earningsCents / 100).toFixed(2);
  await send({
    to,
    subject: `Your beat "${beatTitle}" just sold! 💰`,
    html: `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;background:#0f0f0f;color:#fff;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:28px;font-weight:900;letter-spacing:4px;color:#ff4444">BEATHOLE</span>
      </div>
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:48px">💰</span>
      </div>
      <h2 style="color:#4caf50;text-align:center;margin-bottom:8px">Beat Sold!</h2>
      <p style="color:#aaa;text-align:center">Hey ${creatorName}, you just made a sale!</p>
      <div style="background:#1a1a1a;border:1px solid #4caf5033;border-radius:8px;padding:20px;margin:24px 0">
        <table style="width:100%;color:#aaa;font-size:15px">
          <tr><td>Beat</td><td style="text-align:right;color:#fff;font-weight:700">${beatTitle}</td></tr>
          <tr><td>License</td><td style="text-align:right;color:#fff">${licenseType}</td></tr>
          <tr><td>Buyer</td><td style="text-align:right;color:#fff">@${buyerUsername}</td></tr>
          <tr style="border-top:1px solid #333"><td style="padding-top:12px;color:#4caf50;font-weight:700">Your earnings</td><td style="text-align:right;padding-top:12px;color:#4caf50;font-weight:900;font-size:20px">€${earnings}</td></tr>
        </table>
      </div>
      <div style="text-align:center;margin:28px 0">
        <a href="${FRONTEND}/dashboard" style="background:#ff4444;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px">
          View Dashboard
        </a>
      </div>
      <p style="color:#666;font-size:12px;text-align:center">BeatHole · AI Beat Generation Platform</p>
    </div>`,
  });
}

// ─── Buyer Receipt Email ───────────────────────────────────────────────────────
async function sendBuyerReceiptEmail({ to, buyerName, beatTitle, licenseType, licenseFeatures, amountCents, creatorUsername, beatId }) {
  const amount = (amountCents / 100).toFixed(2);
  const studioLink = `${FRONTEND}/studio/${beatId}`;
  const features = Array.isArray(licenseFeatures) ? licenseFeatures : (licenseFeatures ? JSON.parse(licenseFeatures) : []);
  const featuresList = features.map((f) => `<li style="color:#aaa">${f}</li>`).join('');
  await send({
    to,
    subject: `Your purchase: "${beatTitle}" 🎵`,
    html: `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;background:#0f0f0f;color:#fff;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:28px;font-weight:900;letter-spacing:4px;color:#ff4444">BEATHOLE</span>
      </div>
      <h2 style="color:#fff;text-align:center;margin-bottom:8px">Purchase Confirmed 🎉</h2>
      <p style="color:#aaa;text-align:center">Hey ${buyerName}, your beat is ready!</p>
      <div style="background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:20px;margin:24px 0">
        <table style="width:100%;color:#aaa;font-size:15px">
          <tr><td>Beat</td><td style="text-align:right;color:#fff;font-weight:700">${beatTitle}</td></tr>
          <tr><td>Creator</td><td style="text-align:right;color:#fff">@${creatorUsername}</td></tr>
          <tr><td>License</td><td style="text-align:right;color:#fff">${licenseType}</td></tr>
          <tr style="border-top:1px solid #333"><td style="padding-top:12px;font-weight:700">Total paid</td><td style="text-align:right;padding-top:12px;font-weight:900;font-size:18px">€${amount}</td></tr>
        </table>
        ${featuresList ? `<div style="margin-top:16px"><p style="color:#aaa;margin:0 0 8px;font-size:13px">License includes:</p><ul style="margin:0;padding-left:20px;line-height:1.8">${featuresList}</ul></div>` : ''}
      </div>
      <div style="text-align:center;margin:28px 0">
        <a href="${studioLink}" style="background:#ff4444;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px">
          Open in Studio
        </a>
      </div>
      <p style="color:#666;font-size:13px;text-align:center">Find your beat in <a href="${FRONTEND}/library" style="color:#ff4444">My Library</a></p>
      <p style="color:#666;font-size:12px;text-align:center">BeatHole · AI Beat Generation Platform</p>
    </div>`,
  });
}

// ─── Invite Accepted Email (to inviter) ───────────────────────────────────────
async function sendInviteAcceptedEmail({ to, inviterName, inviteeName, beatTitle, beatId }) {
  const link = `${FRONTEND}/studio/${beatId}`;
  await send({
    to,
    subject: `${inviteeName} accepted your studio invitation 🎛️`,
    html: `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;background:#0f0f0f;color:#fff;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:28px;font-weight:900;letter-spacing:4px;color:#ff4444">BEATHOLE</span>
      </div>
      <h2 style="color:#fff;margin-bottom:8px">Collaboration Accepted! 🤝</h2>
      <p style="color:#aaa">Hey ${inviterName},</p>
      <p style="color:#aaa"><strong style="color:#fff">${inviteeName}</strong> accepted your invitation to collaborate on:</p>
      <div style="background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:16px;margin:20px 0;text-align:center">
        <span style="font-size:20px;font-weight:700;color:#ff4444">🎵 ${beatTitle}</span>
      </div>
      <div style="text-align:center;margin:28px 0">
        <a href="${link}" style="background:#ff4444;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px">
          Open Studio
        </a>
      </div>
      <p style="color:#666;font-size:12px;text-align:center">BeatHole · AI Beat Generation Platform</p>
    </div>`,
  });
}

// ─── Forgot Password Email ────────────────────────────────────────────────────
async function sendForgotPasswordEmail({ to, displayName, resetLink }) {
  await send({
    to,
    subject: 'Reset your BeatHole password',
    html: `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;background:#0f0f0f;color:#fff;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:28px;font-weight:900;letter-spacing:4px;color:#ff4444">BEATHOLE</span>
      </div>
      <h2 style="color:#fff;text-align:center;margin-bottom:8px">Reset Your Password</h2>
      <p style="color:#aaa;text-align:center">Hey ${displayName},</p>
      <p style="color:#aaa;text-align:center">We received a request to reset your password. Click the button below to set a new one. This link expires in 1 hour.</p>
      <div style="text-align:center;margin:28px 0">
        <a href="${resetLink}" style="background:#ff4444;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px">
          Reset Password
        </a>
      </div>
      <p style="color:#666;font-size:13px;text-align:center">If you didn't request this, you can safely ignore this email.</p>
      <p style="color:#666;font-size:12px;text-align:center">BeatHole · AI Beat Generation Platform</p>
    </div>`,
  });
}

// ─── Email Change Confirmation ────────────────────────────────────────────────
async function sendEmailChangeConfirmEmail({ to, displayName, newEmail, confirmLink }) {
  await send({
    to,
    subject: 'Confirm your email change on BeatHole',
    html: `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;background:#0f0f0f;color:#fff;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:28px;font-weight:900;letter-spacing:4px;color:#ff4444">BEATHOLE</span>
      </div>
      <h2 style="color:#fff;text-align:center;margin-bottom:8px">Confirm Email Change</h2>
      <p style="color:#aaa;text-align:center">Hey ${displayName},</p>
      <p style="color:#aaa;text-align:center">You requested to change your email address to:</p>
      <div style="background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:14px;margin:16px 0;text-align:center">
        <span style="color:#fff;font-weight:700">${newEmail}</span>
      </div>
      <p style="color:#aaa;text-align:center">Click the button below to confirm. This link expires in 1 hour.</p>
      <div style="text-align:center;margin:28px 0">
        <a href="${confirmLink}" style="background:#ff4444;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px">
          Confirm Email Change
        </a>
      </div>
      <p style="color:#666;font-size:13px;text-align:center">If you didn't request this, please secure your account immediately.</p>
      <p style="color:#666;font-size:12px;text-align:center">BeatHole · AI Beat Generation Platform</p>
    </div>`,
  });
}

// ─── Password Change Confirmation ─────────────────────────────────────────────
async function sendPasswordChangeConfirmEmail({ to, displayName, confirmLink }) {
  await send({
    to,
    subject: 'Confirm your password change on BeatHole',
    html: `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;background:#0f0f0f;color:#fff;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:28px;font-weight:900;letter-spacing:4px;color:#ff4444">BEATHOLE</span>
      </div>
      <h2 style="color:#fff;text-align:center;margin-bottom:8px">Confirm Password Change</h2>
      <p style="color:#aaa;text-align:center">Hey ${displayName},</p>
      <p style="color:#aaa;text-align:center">You requested to change your password. Click the button below to confirm the change. This link expires in 1 hour.</p>
      <div style="text-align:center;margin:28px 0">
        <a href="${confirmLink}" style="background:#ff4444;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px">
          Confirm Password Change
        </a>
      </div>
      <p style="color:#666;font-size:13px;text-align:center">If you didn't request this, please secure your account immediately.</p>
      <p style="color:#666;font-size:12px;text-align:center">BeatHole · AI Beat Generation Platform</p>
    </div>`,
  });
}

// ─── 2FA Code Email ───────────────────────────────────────────────────────────
async function send2FACodeEmail({ to, displayName, code }) {
  await send({
    to,
    subject: `Your BeatHole login code: ${code}`,
    html: `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;background:#0f0f0f;color:#fff;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:28px;font-weight:900;letter-spacing:4px;color:#ff4444">BEATHOLE</span>
      </div>
      <h2 style="color:#fff;text-align:center;margin-bottom:8px">Your Login Code</h2>
      <p style="color:#aaa;text-align:center">Hey ${displayName},</p>
      <p style="color:#aaa;text-align:center">Use the code below to complete your login. It expires in 10 minutes.</p>
      <div style="background:#1a1a1a;border:1px solid #ff444433;border-radius:12px;padding:24px;margin:24px 0;text-align:center">
        <span style="font-size:40px;font-weight:900;letter-spacing:12px;color:#ff4444">${code}</span>
      </div>
      <p style="color:#666;font-size:13px;text-align:center">If you didn't attempt to log in, please secure your account.</p>
      <p style="color:#666;font-size:12px;text-align:center">BeatHole · AI Beat Generation Platform</p>
    </div>`,
  });
}

// ─── Credits Receipt Email ────────────────────────────────────────────────────
async function sendCreditsReceiptEmail({ to, displayName, credits, amountCents }) {
  const amount = (amountCents / 100).toFixed(2);
  await send({
    to,
    subject: `${credits} beat credits added to your account`,
    html: `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:auto;background:#0f0f0f;color:#fff;padding:32px;border-radius:12px">
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:28px;font-weight:900;letter-spacing:4px;color:#ff4444">BEATHOLE</span>
      </div>
      <div style="text-align:center;margin-bottom:24px">
        <span style="font-size:48px">🎵</span>
      </div>
      <h2 style="color:#fff;text-align:center;margin-bottom:8px">Credits Added!</h2>
      <p style="color:#aaa;text-align:center">Hey ${displayName}, your credits are ready to use.</p>
      <div style="background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:20px;margin:24px 0">
        <table style="width:100%;color:#aaa;font-size:15px">
          <tr><td>Credits added</td><td style="text-align:right;color:#fff;font-weight:700">${credits} beats</td></tr>
          <tr style="border-top:1px solid #333"><td style="padding-top:12px;font-weight:700">Total paid</td><td style="text-align:right;padding-top:12px;font-weight:900;font-size:18px">€${amount}</td></tr>
        </table>
      </div>
      <div style="text-align:center;margin:28px 0">
        <a href="${FRONTEND}/generate" style="background:#ff4444;color:#fff;padding:14px 32px;border-radius:8px;text-decoration:none;font-weight:700;font-size:16px">
          Start Generating
        </a>
      </div>
      <p style="color:#666;font-size:12px;text-align:center">BeatHole · AI Beat Generation Platform</p>
    </div>`,
  });
}

module.exports = {
  sendStudioInviteEmail,
  sendProWelcomeEmail,
  sendBeatSoldEmail,
  sendBuyerReceiptEmail,
  sendInviteAcceptedEmail,
  sendForgotPasswordEmail,
  sendEmailChangeConfirmEmail,
  sendPasswordChangeConfirmEmail,
  send2FACodeEmail,
  sendCreditsReceiptEmail,
};
