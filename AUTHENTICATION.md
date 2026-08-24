# Google passkey authentication

VeriTest uses Streamlit OIDC login. Google decides whether the account uses a password, passkey, fingerprint, face unlock, or device PIN. VeriTest never receives biometric data.

1. Create a Google OAuth web application in Google Cloud Console.
2. Add `https://YOUR-APP-NAME.streamlit.app/oauth2callback` as an authorized redirect URI.
3. In Streamlit Community Cloud, open the app menu and choose **Settings** then **Secrets**.
4. Copy `.streamlit/secrets.toml.example` into the secrets editor and replace every placeholder.
5. Set `AUTH_ENABLED = "true"` and save.
6. Restart the app and test sign-in with an approved Google account.

Keep the client secret and cookie secret private. Do not commit them to GitHub. Use a restricted Google OAuth test-user list until the app has institutional approval.