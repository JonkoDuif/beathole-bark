const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { query } = require('../db');
const { authenticate } = require('../middleware/auth');
const { createNotification } = require('../services/notifications');
const { sendStudioInviteEmail, sendInviteAcceptedEmail } = require('../services/email');

const router = express.Router();

// Helper: check if user has studio access to a beat (owner, buyer, or collaborator)
async function hasStudioAccess(beatId, userId) {
  // Owner
  const ownerRes = await query('SELECT id FROM beats WHERE id = ? AND creator_id = ?', [beatId, userId]);
  if (ownerRes.rows.length > 0) return { access: true, role: 'owner' };

  // Collaborator
  const collabRes = await query(
    'SELECT id FROM studio_collaborators WHERE beat_id = ? AND user_id = ?',
    [beatId, userId]
  );
  if (collabRes.rows.length > 0) return { access: true, role: 'collaborator' };

  // Buyer (completed order)
  const buyerRes = await query(
    "SELECT id FROM orders WHERE beat_id = ? AND buyer_id = ? AND status = 'completed'",
    [beatId, userId]
  );
  if (buyerRes.rows.length > 0) return { access: true, role: 'buyer' };

  return { access: false, role: null };
}

// ─── Get studio project (buyers can access even if beat removed) ──────────────
router.get('/:beatId/project', authenticate, async (req, res, next) => {
  try {
    const { beatId } = req.params;
    const { access } = await hasStudioAccess(beatId, req.user.id);
    if (!access) return res.status(403).json({ error: 'No studio access' });

    const beatRes = await query(
      `SELECT b.*, u.username as creator_username, u.display_name as creator_display_name
       FROM beats b JOIN users u ON b.creator_id = u.id
       WHERE b.id = ?`,
      [beatId]
    );

    if (!beatRes.rows[0]) {
      // Beat was hard-deleted but user has access (via order/collab) — return minimal data
      return res.json({ id: beatId, title: '[Deleted Beat]', status: 'deleted', id: beatId });
    }

    res.json(beatRes.rows[0]);
  } catch (err) {
    next(err);
  }
});

// ─── Version History ──────────────────────────────────────────────────────────

// Save new version
router.post('/:beatId/versions', authenticate, async (req, res, next) => {
  try {
    const { beatId } = req.params;
    const { project_data, label } = req.body;
    if (!project_data) return res.status(400).json({ error: 'project_data is required' });

    const { access } = await hasStudioAccess(beatId, req.user.id);
    if (!access) return res.status(403).json({ error: 'No studio access' });

    // Get next version number
    const countRes = await query(
      'SELECT COUNT(*) as cnt FROM studio_versions WHERE beat_id = ?',
      [beatId]
    );
    const nextVersion = (countRes.rows[0]?.cnt || 0) + 1;

    const id = uuidv4();
    await query(
      'INSERT INTO studio_versions (id, beat_id, saved_by, version_number, label, project_data) VALUES (?, ?, ?, ?, ?, ?)',
      [id, beatId, req.user.id, nextVersion, label || null, JSON.stringify(project_data)]
    );

    // Also update the beat's studio_project (latest)
    await query('UPDATE beats SET studio_project = ? WHERE id = ?', [JSON.stringify(project_data), beatId]);

    // Notify collaborators about new version
    const beatRes = await query('SELECT title, creator_id FROM beats WHERE id = ?', [beatId]);
    const beatTitle = beatRes.rows[0]?.title || 'Untitled';

    const collabRes = await query(
      'SELECT user_id FROM studio_collaborators WHERE beat_id = ? AND user_id != ?',
      [beatId, req.user.id]
    );
    for (const collab of collabRes.rows) {
      await createNotification(collab.user_id, {
        type: 'version_saved',
        title: `New version saved`,
        body: `New version of "${beatTitle}" saved by ${req.user.username || req.user.display_name}`,
        link: `/studio/${beatId}`,
        meta: { beat_id: beatId, version: nextVersion },
      });
    }
    // Also notify owner if collaborator saved
    if (beatRes.rows[0]?.creator_id && beatRes.rows[0].creator_id !== req.user.id) {
      await createNotification(beatRes.rows[0].creator_id, {
        type: 'version_saved',
        title: `New version saved`,
        body: `New version of "${beatTitle}" saved by ${req.user.username || req.user.display_name}`,
        link: `/studio/${beatId}`,
        meta: { beat_id: beatId, version: nextVersion },
      });
    }

    res.json({ id, version_number: nextVersion });
  } catch (err) {
    next(err);
  }
});

// List versions for a beat
router.get('/:beatId/versions', authenticate, async (req, res, next) => {
  try {
    const { beatId } = req.params;
    const { access } = await hasStudioAccess(beatId, req.user.id);
    if (!access) return res.status(403).json({ error: 'No studio access' });

    const versionsRes = await query(
      `SELECT sv.id, sv.version_number, sv.label, sv.created_at,
              u.username as saved_by_username, u.display_name as saved_by_display_name
       FROM studio_versions sv
       JOIN users u ON sv.saved_by = u.id
       WHERE sv.beat_id = ?
       ORDER BY sv.version_number DESC`,
      [beatId]
    );

    res.json(versionsRes.rows);
  } catch (err) {
    next(err);
  }
});

// Get specific version data
router.get('/:beatId/versions/:versionId', authenticate, async (req, res, next) => {
  try {
    const { beatId, versionId } = req.params;
    const { access } = await hasStudioAccess(beatId, req.user.id);
    if (!access) return res.status(403).json({ error: 'No studio access' });

    const versionRes = await query(
      `SELECT sv.*, u.username as saved_by_username
       FROM studio_versions sv
       JOIN users u ON sv.saved_by = u.id
       WHERE sv.id = ? AND sv.beat_id = ?`,
      [versionId, beatId]
    );
    if (!versionRes.rows[0]) return res.status(404).json({ error: 'Version not found' });

    const version = versionRes.rows[0];
    version.project_data = typeof version.project_data === 'string'
      ? JSON.parse(version.project_data)
      : version.project_data;

    res.json(version);
  } catch (err) {
    next(err);
  }
});

// ─── Studio Invitations ───────────────────────────────────────────────────────

// Send invitation
router.post('/:beatId/invite', authenticate, async (req, res, next) => {
  try {
    const { beatId } = req.params;
    const { username } = req.body;
    if (!username) return res.status(400).json({ error: 'username is required' });

    // Only owner can invite
    const beatRes = await query('SELECT id, title, creator_id FROM beats WHERE id = ?', [beatId]);
    if (!beatRes.rows[0]) return res.status(404).json({ error: 'Beat not found' });
    if (beatRes.rows[0].creator_id !== req.user.id) return res.status(403).json({ error: 'Only the owner can invite collaborators' });

    // Pro-only feature
    const inviterUser = await query('SELECT subscription_plan, subscription_status FROM users WHERE id = ?', [req.user.id]);
    const inviter = inviterUser.rows[0];
    const inviterIsPro = inviter?.subscription_plan === 'pro' && inviter?.subscription_status === 'active';
    if (!inviterIsPro) {
      return res.status(403).json({ error: 'Studio collaboration requires a Pro subscription', requiresPro: true });
    }

    // Find invitee
    const inviteeRes = await query('SELECT id, email, username, display_name FROM users WHERE username = ?', [username]);
    if (!inviteeRes.rows[0]) return res.status(404).json({ error: 'User not found' });
    const invitee = inviteeRes.rows[0];
    if (invitee.id === req.user.id) return res.status(400).json({ error: 'Cannot invite yourself' });

    // Check already collaborator
    const existingCollab = await query(
      'SELECT id FROM studio_collaborators WHERE beat_id = ? AND user_id = ?',
      [beatId, invitee.id]
    );
    if (existingCollab.rows.length > 0) return res.status(409).json({ error: 'User is already a collaborator' });

    // Upsert invitation
    const id = uuidv4();
    await query(
      `INSERT INTO studio_invitations (id, beat_id, inviter_id, invitee_id, status)
       VALUES (?, ?, ?, ?, 'pending')
       ON DUPLICATE KEY UPDATE id = id, status = 'pending', updated_at = NOW()`,
      [id, beatId, req.user.id, invitee.id]
    );

    // Notification
    await createNotification(invitee.id, {
      type: 'studio_invite',
      title: 'Studio invitation',
      body: `${req.user.display_name || req.user.username} invited you to collaborate on "${beatRes.rows[0].title}"`,
      link: `/studio/invite/${beatId}`,
      meta: { beat_id: beatId, inviter: req.user.username, beat_title: beatRes.rows[0].title },
    });

    // Email
    try {
      await sendStudioInviteEmail({
        to: invitee.email,
        inviteeName: invitee.display_name || invitee.username,
        inviterName: req.user.display_name || req.user.username,
        beatTitle: beatRes.rows[0].title,
        beatId,
      });
    } catch (e) {
      console.error('Email send failed (non-fatal):', e.message);
    }

    res.json({ message: 'Invitation sent', invitee: invitee.username });
  } catch (err) {
    next(err);
  }
});

// Accept invitation
router.post('/:beatId/invite/accept', authenticate, async (req, res, next) => {
  try {
    const { beatId } = req.params;

    const inviteRes = await query(
      "SELECT id, inviter_id FROM studio_invitations WHERE beat_id = ? AND invitee_id = ? AND status = 'pending'",
      [beatId, req.user.id]
    );
    if (!inviteRes.rows[0]) return res.status(404).json({ error: 'No pending invitation found' });

    // Mark accepted
    await query(
      "UPDATE studio_invitations SET status = 'accepted' WHERE beat_id = ? AND invitee_id = ?",
      [beatId, req.user.id]
    );

    // Add as collaborator
    const scId = uuidv4();
    await query(
      `INSERT IGNORE INTO studio_collaborators (id, beat_id, user_id, invited_by)
       VALUES (?, ?, ?, ?)`,
      [scId, beatId, req.user.id, inviteRes.rows[0].inviter_id]
    );

    // Notify inviter
    const beatRes = await query('SELECT title FROM beats WHERE id = ?', [beatId]);
    const inviterId = inviteRes.rows[0].inviter_id;
    await createNotification(inviterId, {
      type: 'invite_accepted',
      title: 'Invitation accepted',
      body: `${req.user.display_name || req.user.username} accepted your invitation to "${beatRes.rows[0]?.title}"`,
      link: `/studio/${beatId}`,
      meta: { beat_id: beatId },
    });

    // Email inviter
    try {
      const inviterRes = await query('SELECT email, display_name, username FROM users WHERE id = ?', [inviterId]);
      const inviter = inviterRes.rows[0];
      if (inviter) {
        await sendInviteAcceptedEmail({
          to: inviter.email,
          inviterName: inviter.display_name || inviter.username,
          inviteeName: req.user.display_name || req.user.username,
          beatTitle: beatRes.rows[0]?.title || 'Untitled',
          beatId,
        });
      }
    } catch (e) { console.error('Invite accepted email failed (non-fatal):', e.message); }

    res.json({ message: 'Invitation accepted', beatId });
  } catch (err) {
    next(err);
  }
});

// Decline invitation
router.post('/:beatId/invite/decline', authenticate, async (req, res, next) => {
  try {
    const { beatId } = req.params;

    const inviteRes = await query(
      "SELECT id FROM studio_invitations WHERE beat_id = ? AND invitee_id = ? AND status = 'pending'",
      [beatId, req.user.id]
    );
    if (!inviteRes.rows[0]) return res.status(404).json({ error: 'No pending invitation found' });

    await query(
      "UPDATE studio_invitations SET status = 'declined' WHERE beat_id = ? AND invitee_id = ?",
      [beatId, req.user.id]
    );

    res.json({ message: 'Invitation declined' });
  } catch (err) {
    next(err);
  }
});

// Get my invitations (pending)
router.get('/invitations/pending', authenticate, async (req, res, next) => {
  try {
    const res2 = await query(
      `SELECT si.id, si.beat_id, si.created_at,
              b.title as beat_title,
              u.username as inviter_username, u.display_name as inviter_display_name, u.avatar_url as inviter_avatar
       FROM studio_invitations si
       JOIN beats b ON si.beat_id = b.id
       JOIN users u ON si.inviter_id = u.id
       WHERE si.invitee_id = ? AND si.status = 'pending'
       ORDER BY si.created_at DESC`,
      [req.user.id]
    );
    res.json(res2.rows);
  } catch (err) {
    next(err);
  }
});

// Get collaborators for a beat
router.get('/:beatId/collaborators', authenticate, async (req, res, next) => {
  try {
    const { beatId } = req.params;
    const { access } = await hasStudioAccess(beatId, req.user.id);
    if (!access) return res.status(403).json({ error: 'No studio access' });

    const res2 = await query(
      `SELECT u.id, u.username, u.display_name, u.avatar_url, sc.accepted_at
       FROM studio_collaborators sc
       JOIN users u ON sc.user_id = u.id
       WHERE sc.beat_id = ?`,
      [beatId]
    );
    res.json(res2.rows);
  } catch (err) {
    next(err);
  }
});

// Remove (kick) a collaborator — owner only
router.delete('/:beatId/collaborators/:userId', authenticate, async (req, res, next) => {
  try {
    const { beatId, userId } = req.params;
    const beatRes = await query('SELECT creator_id FROM beats WHERE id = ?', [beatId]);
    if (!beatRes.rows[0]) return res.status(404).json({ error: 'Beat not found' });
    if (beatRes.rows[0].creator_id !== req.user.id) {
      return res.status(403).json({ error: 'Only the owner can remove collaborators' });
    }
    await query('DELETE FROM studio_collaborators WHERE beat_id = ? AND user_id = ?', [beatId, userId]);
    await query("UPDATE studio_invitations SET status = 'removed' WHERE beat_id = ? AND invitee_id = ?", [beatId, userId]);
    res.json({ message: 'Collaborator removed' });
  } catch (err) {
    next(err);
  }
});

// Get beats where I am a collaborator (for My Beats)
router.get('/my-collabs', authenticate, async (req, res, next) => {
  try {
    const res2 = await query(
      `SELECT b.*, u.username as creator_username, u.display_name as creator_display_name, u.avatar_url as creator_avatar
       FROM studio_collaborators sc
       JOIN beats b ON sc.beat_id = b.id
       JOIN users u ON b.creator_id = u.id
       WHERE sc.user_id = ?
       ORDER BY sc.accepted_at DESC`,
      [req.user.id]
    );
    res.json(res2.rows);
  } catch (err) {
    next(err);
  }
});

module.exports = router;
