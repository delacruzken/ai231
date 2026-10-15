# Spotify playback (PLAY_MUSIC)

When a voice command accepts `PLAY_MUSIC`, ME2 calls the [Spotify Web API](https://developer.spotify.com/documentation/web-api) to **shuffle-play** your playlist on the **Mac Spotify desktop app** (Spotify Connect).

Playlist default: [güdmüd](https://open.spotify.com/playlist/3eurayPQQ7IvoqkkletA89) → `spotify:playlist:3eurayPQQ7IvoqkkletA89`

## One-time setup

1. Spotify Developer Dashboard → your app → **Settings**
2. Add Redirect URI exactly: `http://127.0.0.1:8888/callback`
3. Copy Client ID + Client Secret into env file:

```bash
cd me2
cp .env.example .env
# edit .env — fill SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
```

4. Authorize once (opens browser, Premium account):

```bash
cd me2
python -m scripts.spotify_auth
```

Tokens are saved to `me2/.spotify_tokens.json` (gitignored).

5. Open the **Mac Spotify app** (play any song once so it becomes a Connect device).

6. Smoke test:

```bash
python -m scripts.spotify_auth --test-play
```

## Voice demo

Run the assistant as usual (`--demo-ui` optional). After wake → **play music**:

- Den vinyl spins (UI)
- Mac Spotify starts shuffled playback on that playlist

Also mapped: `PAUSE` / `STOP` → pause, `NEXT` → next track, `VOLUME_UP` / `VOLUME_DOWN` → ±10%.

**Alexa-style ducking:** on wake while Spotify is playing, volume drops to **20%** of the current level. It **stays low** while the den speaks confirmations (weather, time, lights, reject cue, etc.), then restores. `PLAY_MUSIC` / `PAUSE` / `STOP` restore immediately around that action so playback isn’t left ducked.

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| `no Spotify Connect devices` | Open Mac Spotify app; play something manually once |
| `not authorized` | Re-run `python -m scripts.spotify_auth` |
| 403 Premium | Account must be Premium for player endpoints |
| Wrong playlist | Set `SPOTIFY_PLAYLIST_URI` in `.env` |
