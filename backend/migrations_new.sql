-- Auth tokens table (forgot password, email change verification, 2FA)
CREATE TABLE IF NOT EXISTS auth_tokens (
  id VARCHAR(36) PRIMARY KEY,
  user_id VARCHAR(36) NOT NULL,
  type ENUM('forgot_password', 'verify_email_change', 'verify_password_change', '2fa_code') NOT NULL,
  token VARCHAR(64) NOT NULL UNIQUE,
  data JSON NULL,
  expires_at DATETIME NOT NULL,
  used TINYINT(1) DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_token (token),
  INDEX idx_user_type (user_id, type)
);

-- 2FA settings on users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS two_fa_enabled TINYINT(1) DEFAULT 0;
