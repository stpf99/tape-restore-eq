# Audio EQ Ultimate v8.0 - Documentation

## Overview

Advanced real-time audio equalizer with AI-powered presets, intelligent silence detection, and comprehensive processing logging.

## Key Features

### 1. AI Preset System
- **20+ Technical Presets**: Including Studio Monitor, Mastering Reference, genre-specific (Rock, Jazz, EDM, Classical, etc.)
- **Auto-Selection**: AI automatically selects the best preset based on audio analysis
- **Custom Presets**: Save your own EQ curves with descriptions and tags
- **Categories**:
  - Reference (Flat, Studio Monitor, Mastering)
  - Genre-specific (Rock/Metal, Electronic/EDM, Jazz/Acoustic, Classical, Hip-Hop, etc.)
  - Creative/Effect (Bass Boost, Treble Boost, Mid Scoop, Warm Analog, Bright Digital)
  - Correction (De-Harsh, De-Mud, Presence Boost, Air Enhancement)

### 2. Silence Detection & Bypass
- **Smart Detection**: Automatically detects silence periods (default: <-50dB for >0.3s)
- **Processing Bypass**: Skips CPU-intensive processing during silence
- **Duration Tracking**: Shows how long audio has been silent
- **Configurable**: Adjustable threshold and minimum duration

### 3. Real-Time Processing Log
- **Comprehensive Logging**: Tracks all EQ changes, normalization adjustments, preset changes
- **Timestamped Entries**: Millisecond precision timestamps
- **Export Function**: Save logs as JSON for analysis
- **Log Types**:
  - `EQ_CHANGE`: Band adjustments (manual, auto-EQ, preset)
  - `NORMALIZATION`: Gain adjustments with LUFS/Peak values
  - `SILENCE_BYPASS`: Silence detection events
  - `PRESET_CHANGE`: Manual or AI preset selection
  - `AUTO_EQ`: Auto-EQ toggle events
  - `IMPERFECTION`: Detected audio issues (resonance, muddiness, harshness)

### 4. Advanced Audio Processing
- **10-Band Equalizer**: Precise control at 31, 62, 125, 250, 500, 1k, 2k, 4k, 8k, 16kHz
- **Smart Auto-EQ**: Analyzes spectrum and applies intelligent corrections
- **Normalization**: LUFS-based with anti-blast protection
- **Spectrum Analysis**: Real-time frequency visualization
- **Imperfection Detection**: Identifies resonance, muddiness, harshness

### 5. Source Selection
- **Multiple Sources**: Monitor any audio output or input device
- **PulseAudio Integration**: Lists all available sinks and sources
- **Hot-Swapping**: Change source without restarting

## Installation

### Requirements
```bash
# Debian/Ubuntu
sudo apt install python3-gi gstreamer1.0-tools gstreamer1.0-plugins-good \
                 gstreamer1.0-plugins-bad python3-numpy pulseaudio

# Arch Linux
sudo pacman -S python-gobject gstreamer gst-plugins-good gst-plugins-bad \
               python-numpy pulseaudio
```

### Running
```bash
python3 audio_eq_ultimate.py
```

## Usage Guide

### Getting Started
1. **Launch Application**: Run the script
2. **Select Source**: Choose audio source from dropdown (default: autoeq_sink monitor)
3. **Start Listening**: Click "▶ Start Listening"
4. **Choose Preset**: Select from AI presets or create custom
5. **Enable Features**: Toggle Auto-EQ, Normalization, Silence Bypass as needed

### AI Preset Auto-Selection
1. Enable "Auto-Select Preset (AI)" checkbox
2. Application analyzes audio characteristics:
   - Bass/Mid/High energy distribution
   - Detected imperfections (harshness, muddiness, resonance)
3. Automatically selects optimal preset
4. Logged in RT Processing Log

### Manual EQ Adjustment
- **Sliders**: Adjust individual bands (-12dB to +12dB)
- **Reset**: Click "Reset EQ" to return to flat
- **Save**: Create custom preset from current settings

### Monitoring
- **Spectrum**: Real-time frequency analysis visualization
- **Level Meters**: LUFS and Peak meters with color coding
  - Green: Safe (-70 to -14 LUFS, -70 to -6 Peak)
  - Yellow: Caution (-14 to -6 LUFS, -6 to -1 Peak)
  - Red: Warning (>-6 LUFS, >-1 Peak)
- **Processing Log**: Scrolling log of all RT operations

### Silence Bypass
- **Automatic**: Enabled by default
- **Threshold**: -50dB (audio below this is considered silence)
- **Duration**: 0.3s minimum before bypass activates
- **Status**: Shows "Active" or "Bypassed (X.Xs)"
- **Benefits**: Reduces CPU usage, prevents processing of noise floor

### Export Log
1. Click "Export Log"
2. Saves to `eq_log_YYYYMMDD_HHMMSS.json`
3. Contains:
   - Current state (EQ gains, normalization, active features)
   - Complete log history (up to 200 entries)

## Technical Presets Reference

### Reference Presets
- **Flat Reference**: Pure monitoring, no coloration
- **Studio Monitor**: Slight presence boost for mixing
- **Mastering Reference**: Enhanced clarity for final checks

### Genre Presets
- **Rock/Metal**: Punchy low-end, aggressive mids, controlled highs
- **Electronic/EDM**: Deep bass, scooped mids, crisp highs
- **Jazz/Acoustic**: Natural warmth with vocal presence
- **Classical/Orchestral**: Wide soundstage, natural dynamics
- **Hip-Hop/R&B**: Deep bass, smooth vocals
- **Vocal Enhancement**: Vocal clarity and presence
- **Podcast/Speech**: Optimized speech intelligibility

### Effect Presets
- **Bass Boost**: Maximum low-end emphasis
- **Treble Boost**: Maximum high-end emphasis
- **Mid Scoop**: Modern scooped sound
- **Warm Analog**: Vintage warm character
- **Bright Digital**: Modern digital brightness

### Correction Presets
- **De-Harsh**: Reduces harshness and sibilance
- **De-Mud**: Cleans up muddy low-end
- **Presence Boost**: Enhances vocal presence
- **Air Enhancement**: Adds air and sparkle

## Processing Log Examples

```
[14:23:45.123] PRESET: Rock/Metal (AUTO)
[14:23:45.234] EQ: 62Hz +0.0 → +2.5dB (Preset: Rock/Metal)
[14:23:45.345] EQ: 125Hz +0.0 → +1.5dB (Preset: Rock/Metal)
[14:23:46.456] NORM: +3.2dB (LUFS: -17.2, Peak: -4.3)
[14:23:47.567] SILENCE: ENABLED (0.5s)
[14:23:50.678] SILENCE: DISABLED
[14:23:51.789] ISSUE: mid-high - harshness (sev: 1.2, corr: -3.0dB)
[14:23:51.890] AUTO-EQ: ON
[14:23:52.012] EQ: 4000Hz +1.5 → -1.2dB (Auto-EQ)
```

## Advanced Configuration

### Silence Detector
Modify in code:
```python
self.silence_detector = SilenceDetector(
    threshold_db=-50,    # dB threshold
    min_duration=0.3     # seconds before bypass
)
```

### Normalization
```python
self.normalizer.target_lufs = -14.0      # Target loudness
self.normalizer.max_peak = -0.5          # Peak limit
self.normalizer.max_gain_change = 0.5    # dB per update (anti-blast)
```

### Auto-EQ Sensitivity
```python
self.smoothing_factor = 0.15  # Lower = slower response
```

### Log Size
```python
self.rt_log = RTProcessingLog(max_entries=200)  # Max log entries
```

## Troubleshooting

### No Audio
1. Check PulseAudio is running: `pulseaudio --check`
2. Verify source selection
3. Check volume levels aren't at 0

### High CPU Usage
1. Enable "Silence Bypass"
2. Disable "Auto-EQ" when not needed
3. Reduce spectrum update rate (modify code: `interval` property)

### Presets Not Saving
1. Check permissions on `~/.config/`
2. Verify disk space available

### Processing Too Aggressive
1. Lower Auto-EQ sensitivity (reduce `smoothing_factor`)
2. Use gentler presets (e.g., "Flat Reference")
3. Disable normalization

## Performance Tips

1. **Silence Bypass**: Always keep enabled for efficiency
2. **Auto-Preset**: Disable if using manual EQ
3. **Export Logs**: Periodically export to prevent memory buildup
4. **Source Selection**: Monitor specific apps via their sink monitors

## FAQ

**Q: What's the difference between Auto-EQ and AI Preset selection?**
A: Auto-EQ continuously adjusts EQ bands based on spectrum analysis. AI Preset selection picks the best preset from the library based on audio characteristics.

**Q: Can I use both Auto-EQ and presets together?**
A: Yes! Select a preset as a starting point, then enable Auto-EQ for fine-tuning.

**Q: What does "Silence Bypass" do exactly?**
A: It stops processing audio when input is below -50dB for >0.3s, reducing CPU usage.

**Q: How do I monitor specific applications?**
A: Select the application's monitor (e.g., "Monitor: Firefox") from the source dropdown.

**Q: Can I create genre-specific auto-presets?**
A: Yes! The AI analyzes bass/mid/high ratios and imperfections to auto-select genre presets.

**Q: What's the processing log for?**
A: It helps debug issues, understand what processing is being applied, and analyze audio over time.

## Credits

Audio EQ Ultimate v8.0
- Advanced real-time audio processing
- AI-powered preset management
- Intelligent silence detection
- Comprehensive processing logging

Built with:
- Python 3
- GTK 3
- GStreamer
- NumPy
- PulseAudio

## License

GPL-3.0 License - Feel free to modify and distribute.
