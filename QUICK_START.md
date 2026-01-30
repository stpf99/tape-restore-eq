# Audio EQ Ultimate v8.0 - Quick Start Guide

## 🚀 Quick Installation

```bash
# Install dependencies (Debian/Ubuntu)
sudo apt install python3-gi gstreamer1.0-tools gstreamer1.0-plugins-good \
                 gstreamer1.0-plugins-bad python3-numpy pulseaudio

# Run
python3 audio_eq_ultimate.py
```

## 🎵 Basic Usage (5 steps)

1. **Click "▶ Start Listening"**
2. **Select a Preset** (e.g., "Rock/Metal" for music, "Podcast/Speech" for voice)
3. **Click "Apply"**
4. **Optional**: Enable "Smart Balance (Auto-EQ)" for adaptive processing
5. **Optional**: Enable "Normalization (Safe)" for volume leveling

## 🎛️ Key Features

### AI Presets (20+ included)
- **Reference**: Flat, Studio Monitor, Mastering
- **Music**: Rock, Jazz, EDM, Classical, Hip-Hop
- **Voice**: Vocal Enhancement, Podcast
- **Effects**: Bass/Treble Boost, Warm/Bright
- **Fix**: De-Harsh, De-Mud, Presence Boost

### Smart Features
- ✅ **Silence Bypass**: Auto-skips processing during silence (saves CPU)
- ✅ **Auto-Preset Selection**: AI picks best preset for your audio
- ✅ **Real-Time Log**: See every processing change
- ✅ **Anti-Blast Protection**: Safe normalization without distortion

## 🔧 Common Workflows

### For Music Listening
```
1. Select preset matching genre (e.g., "Electronic/EDM")
2. Enable "Smart Balance (Auto-EQ)" ✓
3. Enable "Normalization (Safe)" ✓
4. Keep "Silence Bypass" enabled ✓
```

### For Podcasts/Videos
```
1. Select "Podcast/Speech" preset
2. Enable "Normalization (Safe)" ✓
3. Optionally: "Vocal Enhancement" preset for clarity
```

### For Monitoring/Production
```
1. Select "Flat Reference" or "Studio Monitor"
2. Disable Auto-EQ
3. Disable Normalization
4. Manual EQ adjustments as needed
```

### For Problem Audio (harsh, muddy)
```
1. Let it play for a few seconds
2. Check "Processing Log" for detected issues
3. Apply correction preset:
   - Harsh/Sibilant → "De-Harsh"
   - Muddy/Boomy → "De-Mud"
   - Lacking presence → "Presence Boost"
   - Dull → "Air Enhancement"
```

## 📊 What the Meters Mean

### LUFS (Loudness)
- **Green (-70 to -14)**: Normal listening
- **Yellow (-14 to -6)**: Getting loud
- **Red (above -6)**: Too loud!

### Peak
- **Green (-70 to -6)**: Safe
- **Yellow (-6 to -1)**: Caution
- **Red (above -1)**: Clipping risk!

### Status
- **Active**: Processing audio
- **Bypassed (X.Xs)**: Silence detected for X.X seconds

## 🎚️ Manual EQ Tips

Each slider controls a frequency band:
- **31-125 Hz**: Sub-bass, bass
- **250-500 Hz**: Warmth, body
- **1-2 kHz**: Presence, clarity
- **4-8 kHz**: Brilliance, detail
- **16 kHz**: Air, sparkle

**Rule of thumb**: Cut before you boost!

## 💾 Saving Your Settings

1. Adjust EQ to your liking
2. Click "Save Current as Preset"
3. Enter name and description
4. Your preset appears in the list!

## 📋 Processing Log

Shows real-time changes:
```
[Time] EQ: 250Hz +0.0 → +2.0dB (Auto-EQ)
[Time] PRESET: Jazz/Acoustic (AUTO)
[Time] NORM: +1.5dB (LUFS: -16.2)
[Time] SILENCE: ENABLED (0.8s)
```

Click "Export Log" to save for analysis.

## ⚡ Performance Tips

- **Always keep "Silence Bypass" enabled** - Saves CPU during quiet moments
- **Disable Auto-EQ if using manual settings** - Choose one or the other
- **Use presets as starting points** - Then fine-tune manually
- **Export logs periodically** - Prevents memory buildup

## 🎯 Recommended Presets by Use Case

| Use Case | Preset | Notes |
|----------|--------|-------|
| General music | Auto-Select (AI) | Let AI choose |
| Rock/Metal music | Rock/Metal | Punchy, aggressive |
| Electronic music | Electronic/EDM | Deep bass, bright |
| Classical | Classical/Orchestral | Natural, wide |
| Podcasts | Podcast/Speech | Clear voice |
| Gaming | Rock/Metal | Immersive, dynamic |
| Movies | Classical/Orchestral | Cinematic |
| Harsh audio | De-Harsh | Reduces sibilance |
| Muddy audio | De-Mud | Cleans low-end |

## 🆘 Troubleshooting

**No sound?**
1. Check source selection
2. Verify "Start Listening" is clicked
3. Check system volume

**Distorted sound?**
1. Disable normalization
2. Reset EQ to flat
3. Check if Peak meter is red

**CPU high?**
1. Ensure "Silence Bypass" is enabled
2. Disable Auto-EQ
3. Close other applications

**Presets not applying?**
1. Click "Apply" after selecting
2. Check Processing Log for changes

## 🔗 Source Selection

Choose what to process:
- **Monitor: [app]** - Process that app's audio
- **Input: [device]** - Process microphone/line-in
- **autoeq_sink (Default)** - Process all system audio

## 📖 Full Documentation

See `DOCUMENTATION.md` for:
- Complete preset reference
- Advanced configuration
- Processing log format
- Troubleshooting details
- FAQ

## 🎓 Learning Path

1. **Day 1**: Try different presets, see what you like
2. **Day 2**: Enable Auto-EQ, observe log
3. **Day 3**: Try Auto-Preset selection
4. **Day 4**: Make manual adjustments
5. **Day 5**: Save your first custom preset!

---

**Enjoy your enhanced audio!** 🎧
