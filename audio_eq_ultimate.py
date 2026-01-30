#!/usr/bin/env python3
"""
Audio EQ Ultimate v8.0 - Advanced Real-Time Audio Processing
==============================================================
Features:
- AI-powered technical presets with auto-selection
- Smart silence detection and bypass
- Real-time processing log with applied transitions
- Source selection (Input/Monitor)
- Dynamic Auto-EQ with multiple modes
- Advanced normalization with anti-blast protection
- Comprehensive spectrum analysis and visualization
- Imperfection detection (Resonance/Mud/Harshness)
"""

import gi
import subprocess
import numpy as np
import math
import os
import json
from collections import deque
from datetime import datetime
import time

gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Gtk, Gst, GLib, Gdk, Pango, PangoCairo

# Initialize GStreamer
Gst.init(None)

# ==================== UTILITIES ====================

def create_virtual_sink(sink_name="autoeq_sink"):
    """Creates virtual audio output via pactl."""
    try:
        check = subprocess.run(["pactl", "list", "short", "sinks"], 
                             capture_output=True, text=True)
        if sink_name in check.stdout:
            return True
        subprocess.run(
            ["pactl", "load-module", "module-null-sink", f"sink_name={sink_name}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"Created virtual audio sink: {sink_name}")
        return True
    except Exception as e:
        print(f"Info: {e}")
        return False

def get_pulse_sources():
    """Gets list of PulseAudio sources."""
    sources = []
    seen = set()
    
    # 1. Monitors
    try:
        result = subprocess.run(["pactl", "list", "short", "sinks"], 
                              capture_output=True, text=True)
        for line in result.stdout.splitlines():
            parts = line.split('\t')
            if len(parts) >= 2:
                name = parts[1]
                mon = f"{name}.monitor"
                if mon not in seen:
                    sources.append((mon, f"Monitor: {name}"))
                    seen.add(mon)
    except: pass

    # 2. Physical inputs
    try:
        result = subprocess.run(["pactl", "list", "short", "sources"], 
                              capture_output=True, text=True)
        for line in result.stdout.splitlines():
            parts = line.split('\t')
            if len(parts) >= 2:
                name = parts[1]
                if "monitor" not in name and name not in seen:
                    sources.append((name, f"Input: {name}"))
                    seen.add(name)
    except: pass

    sources.sort(key=lambda x: x[1])
    if not any(s[0] == "autoeq_sink.monitor" for s in sources):
        sources.insert(0, ("autoeq_sink.monitor", "Monitor: autoeq_sink (Default)"))
    return sources

# ==================== AI PRESET SYSTEM ====================

class AIPresetManager:
    """AI-powered technical preset manager with auto-selection."""
    
    def __init__(self):
        self.presets = {
            # Technical Presets
            "Flat Reference": {
                "gains": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "description": "Neutral reference - no coloration",
                "tags": ["reference", "neutral", "monitoring"]
            },
            "Studio Monitor": {
                "gains": [0.5, 0.3, 0, 0, 0.2, 0.5, 0.8, 0.5, 0.2, 0],
                "description": "Studio monitoring - slight presence boost",
                "tags": ["monitoring", "studio", "reference"]
            },
            "Mastering Reference": {
                "gains": [1, 0.5, 0, 0, 0, 0, 0.3, 0.8, 1.2, 1.5],
                "description": "High-end clarity for mastering",
                "tags": ["mastering", "reference", "clarity"]
            },
            
            # Genre-specific
            "Rock/Metal": {
                "gains": [3, 2.5, 1.5, 0.5, 0, 0.5, 1.5, 2.5, 2, 1],
                "description": "Punchy low-end, aggressive mids, controlled highs",
                "tags": ["rock", "metal", "aggressive"]
            },
            "Electronic/EDM": {
                "gains": [4, 3, 2, 0, -1, -0.5, 1, 2.5, 3.5, 4],
                "description": "Deep bass, crisp highs, scooped mids",
                "tags": ["electronic", "edm", "bass"]
            },
            "Jazz/Acoustic": {
                "gains": [1.5, 1, 0.5, 0, 0.5, 1, 1.5, 2, 1.5, 0.5],
                "description": "Natural warmth with presence",
                "tags": ["jazz", "acoustic", "natural"]
            },
            "Classical/Orchestral": {
                "gains": [0.5, 0.8, 1.2, 1.5, 1, 0.5, 0.5, 1.5, 2.5, 3],
                "description": "Wide soundstage, natural dynamics",
                "tags": ["classical", "orchestral", "natural"]
            },
            "Hip-Hop/R&B": {
                "gains": [5, 4, 2.5, 1, 0, -0.5, 0.5, 1.5, 2, 2.5],
                "description": "Deep bass, smooth vocals",
                "tags": ["hiphop", "rnb", "bass", "vocals"]
            },
            "Vocal Enhancement": {
                "gains": [0, 0, 0.5, 1.5, 2.5, 3, 2, 1, 0, -1],
                "description": "Vocal clarity and presence",
                "tags": ["vocal", "speech", "clarity"]
            },
            "Podcast/Speech": {
                "gains": [-2, -1, 0, 1, 2.5, 3, 2, 0.5, -1, -2],
                "description": "Speech intelligibility",
                "tags": ["speech", "podcast", "clarity"]
            },
            
            # Creative/Effect
            "Bass Boost": {
                "gains": [6, 5, 4, 2, 0.5, 0, 0, 0, 0, 0],
                "description": "Maximum low-end emphasis",
                "tags": ["bass", "effect"]
            },
            "Treble Boost": {
                "gains": [0, 0, 0, 0, 0, 0, 1.5, 3, 4.5, 6],
                "description": "Maximum high-end emphasis",
                "tags": ["treble", "effect", "bright"]
            },
            "Mid Scoop": {
                "gains": [2, 1.5, 1, 0, -3, -4, -3, 0, 1, 2],
                "description": "Scooped mids for modern sound",
                "tags": ["effect", "modern"]
            },
            "Warm Analog": {
                "gains": [2.5, 2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5, -2],
                "description": "Warm vintage character",
                "tags": ["warm", "vintage", "analog"]
            },
            "Bright Digital": {
                "gains": [-2, -1.5, -1, -0.5, 0, 0.5, 1.5, 2.5, 3.5, 4],
                "description": "Modern digital brightness",
                "tags": ["bright", "digital", "modern"]
            },
            
            # Correction Presets
            "De-Harsh": {
                "gains": [0, 0, 0, 0, 0, -1, -2.5, -3, -2, -1],
                "description": "Reduce harshness and sibilance",
                "tags": ["correction", "deharse"]
            },
            "De-Mud": {
                "gains": [-3, -2.5, -2, -1, 0, 0, 0, 0, 0, 0],
                "description": "Clean up muddy low-end",
                "tags": ["correction", "clarity"]
            },
            "Presence Boost": {
                "gains": [0, 0, 0, 0, 1, 2, 3, 2.5, 1.5, 0.5],
                "description": "Enhance vocal presence",
                "tags": ["presence", "vocal"]
            },
            "Air Enhancement": {
                "gains": [0, 0, 0, 0, 0, 0, 0.5, 1.5, 2.5, 4],
                "description": "Add air and sparkle",
                "tags": ["air", "bright", "clarity"]
            }
        }
        
        self.current_preset = "Flat Reference"
        self.auto_select_enabled = False
        
    def get_preset(self, name):
        """Get preset by name."""
        if name in self.presets:
            return self.presets[name]["gains"]
        return [0] * 10
    
    def get_preset_info(self, name):
        """Get full preset information."""
        return self.presets.get(name, {})
    
    def auto_select_preset(self, band_analysis, imperfections):
        """AI-powered automatic preset selection based on audio analysis."""
        if not self.auto_select_enabled:
            return None
        
        scores = {}
        
        for preset_name, preset_data in self.presets.items():
            score = 0
            
            # Analyze audio characteristics
            bass_energy = sum(band_analysis.get('low', {}).get('energy', 0) for _ in range(1))
            mid_energy = sum(band_analysis.get('mid', {}).get('energy', 0) for _ in range(1))
            high_energy = sum(band_analysis.get('high', {}).get('energy', 0) for _ in range(1))
            
            total_energy = bass_energy + mid_energy + high_energy
            if total_energy == 0:
                continue
            
            # Calculate energy distribution
            bass_ratio = bass_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
            
            # Check for imperfections
            has_harshness = any('harshness' in str(issues) for issues in imperfections.values())
            has_muddiness = any('muddiness' in str(issues) for issues in imperfections.values())
            has_resonance = any('resonance' in str(issues) for issues in imperfections.values())
            
            # Score based on audio characteristics
            if bass_ratio > 0.4:
                if 'bass' in preset_data.get('tags', []):
                    score += 30
            
            if mid_ratio > 0.4:
                if 'vocal' in preset_data.get('tags', []) or 'presence' in preset_data.get('tags', []):
                    score += 30
            
            if high_ratio > 0.3:
                if 'bright' in preset_data.get('tags', []) or 'air' in preset_data.get('tags', []):
                    score += 30
            
            # Score based on imperfections
            if has_harshness and 'deharse' in preset_data.get('tags', []):
                score += 50
            
            if has_muddiness and 'correction' in preset_data.get('tags', []):
                score += 50
            
            scores[preset_name] = score
        
        # Select best preset
        if scores:
            best_preset = max(scores, key=scores.get)
            if scores[best_preset] > 30:  # Threshold for auto-selection
                return best_preset
        
        return None
    
    def save_custom_preset(self, name, gains, description="Custom preset", tags=None):
        """Save a custom preset."""
        if tags is None:
            tags = ["custom"]
        
        self.presets[name] = {
            "gains": list(gains),
            "description": description,
            "tags": tags
        }
        
        # Save to file
        try:
            preset_file = os.path.expanduser("~/.config/audio_eq_presets.json")
            os.makedirs(os.path.dirname(preset_file), exist_ok=True)
            
            with open(preset_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
        except Exception as e:
            print(f"Error saving preset: {e}")
    
    def load_custom_presets(self):
        """Load custom presets from file."""
        try:
            preset_file = os.path.expanduser("~/.config/audio_eq_presets.json")
            if os.path.exists(preset_file):
                with open(preset_file, 'r') as f:
                    loaded = json.load(f)
                    self.presets.update(loaded)
        except Exception as e:
            print(f"Error loading presets: {e}")

# ==================== SILENCE DETECTION ====================

class SilenceDetector:
    """Smart silence detector to bypass processing during quiet periods."""
    
    def __init__(self, threshold_db=-50, min_duration=0.5):
        self.threshold_db = threshold_db
        self.min_duration = min_duration  # seconds
        self.silence_start = None
        self.is_silent = False
        self.sample_rate = 44100
        self.silence_samples = int(min_duration * self.sample_rate)
        self.silence_buffer = deque(maxlen=self.silence_samples)
        
    def update(self, audio_level_db):
        """Update silence detection state."""
        self.silence_buffer.append(audio_level_db)
        
        # Check if currently silent
        if len(self.silence_buffer) == self.silence_samples:
            avg_level = np.mean(self.silence_buffer)
            
            if avg_level < self.threshold_db:
                if not self.is_silent:
                    self.is_silent = True
                    self.silence_start = time.time()
            else:
                if self.is_silent:
                    self.is_silent = False
                    self.silence_start = None
        
        return self.is_silent
    
    def get_silence_duration(self):
        """Get current silence duration in seconds."""
        if self.is_silent and self.silence_start:
            return time.time() - self.silence_start
        return 0.0
    
    def should_bypass(self):
        """Check if processing should be bypassed."""
        return self.is_silent

# ==================== RT PROCESSING LOG ====================

class RTProcessingLog:
    """Real-time processing log for tracking applied audio transitions."""
    
    def __init__(self, max_entries=100):
        self.entries = deque(maxlen=max_entries)
        self.current_state = {
            'eq_gains': [0.0] * 10,
            'normalization_gain': 0.0,
            'auto_eq_active': False,
            'silence_bypassed': False,
            'preset': None
        }
        
    def log_eq_change(self, band_idx, old_value, new_value, reason="Manual"):
        """Log EQ band change."""
        entry = {
            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'type': 'EQ_CHANGE',
            'band': band_idx,
            'freq': [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000][band_idx],
            'old_value': round(old_value, 2),
            'new_value': round(new_value, 2),
            'delta': round(new_value - old_value, 2),
            'reason': reason
        }
        self.entries.append(entry)
        self.current_state['eq_gains'][band_idx] = new_value
        
    def log_normalization(self, gain_db, lufs, peak):
        """Log normalization adjustment."""
        entry = {
            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'type': 'NORMALIZATION',
            'gain_db': round(gain_db, 2),
            'lufs': round(lufs, 2),
            'peak': round(peak, 2)
        }
        self.entries.append(entry)
        self.current_state['normalization_gain'] = gain_db
        
    def log_silence_bypass(self, enabled, duration=0.0):
        """Log silence bypass activation/deactivation."""
        entry = {
            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'type': 'SILENCE_BYPASS',
            'enabled': enabled,
            'duration': round(duration, 2)
        }
        self.entries.append(entry)
        self.current_state['silence_bypassed'] = enabled
        
    def log_preset_change(self, preset_name, auto_selected=False):
        """Log preset change."""
        entry = {
            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'type': 'PRESET_CHANGE',
            'preset': preset_name,
            'auto_selected': auto_selected
        }
        self.entries.append(entry)
        self.current_state['preset'] = preset_name
        
    def log_auto_eq_toggle(self, enabled):
        """Log auto-EQ toggle."""
        entry = {
            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'type': 'AUTO_EQ',
            'enabled': enabled
        }
        self.entries.append(entry)
        self.current_state['auto_eq_active'] = enabled
        
    def log_imperfection_detected(self, band, issue_type, severity, correction):
        """Log detected audio imperfection."""
        entry = {
            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'type': 'IMPERFECTION',
            'band': band,
            'issue': issue_type,
            'severity': round(severity, 2),
            'correction': round(correction, 2)
        }
        self.entries.append(entry)
        
    def get_recent_entries(self, count=20):
        """Get recent log entries."""
        return list(self.entries)[-count:]
    
    def get_formatted_log(self, count=20):
        """Get formatted log string."""
        lines = []
        for entry in self.get_recent_entries(count):
            if entry['type'] == 'EQ_CHANGE':
                lines.append(f"[{entry['timestamp']}] EQ: {entry['freq']}Hz "
                           f"{entry['old_value']:+.1f} → {entry['new_value']:+.1f}dB "
                           f"({entry['reason']})")
            elif entry['type'] == 'NORMALIZATION':
                lines.append(f"[{entry['timestamp']}] NORM: {entry['gain_db']:+.1f}dB "
                           f"(LUFS: {entry['lufs']:.1f}, Peak: {entry['peak']:.1f})")
            elif entry['type'] == 'SILENCE_BYPASS':
                status = "ENABLED" if entry['enabled'] else "DISABLED"
                lines.append(f"[{entry['timestamp']}] SILENCE: {status} "
                           f"({entry['duration']:.1f}s)")
            elif entry['type'] == 'PRESET_CHANGE':
                auto = " (AUTO)" if entry['auto_selected'] else ""
                lines.append(f"[{entry['timestamp']}] PRESET: {entry['preset']}{auto}")
            elif entry['type'] == 'AUTO_EQ':
                status = "ON" if entry['enabled'] else "OFF"
                lines.append(f"[{entry['timestamp']}] AUTO-EQ: {status}")
            elif entry['type'] == 'IMPERFECTION':
                lines.append(f"[{entry['timestamp']}] ISSUE: {entry['band']} - "
                           f"{entry['issue']} (sev: {entry['severity']:.1f}, "
                           f"corr: {entry['correction']:+.1f}dB)")
        
        return "\n".join(lines)
    
    def export_log(self, filename):
        """Export log to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'current_state': self.current_state,
                    'log_entries': list(self.entries)
                }, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting log: {e}")
            return False

# ==================== AUDIO PROCESSING ====================

class AudioNormalizer:
    """Advanced normalization with anti-blast protection."""
    def __init__(self):
        self.target_lufs = -14.0
        self.max_peak = -0.5
        self.current_lufs = -70.0
        self.current_peak = -70.0
        self.smoothing = 0.1
        self.current_gain = 0.0
        self.gain_history = deque(maxlen=20)
        self.blast_protection = True
        self.max_gain_change = 0.5  # dB per update
        
    def calculate_lufs(self, audio_data):
        if len(audio_data) == 0: return -70.0
        rms = np.sqrt(np.mean(audio_data**2))
        return (20 * np.log10(rms) - 0.691) if rms > 0 else -70.0
    
    def calculate_gain(self):
        if self.current_lufs > -70:
            lufs_gain = self.target_lufs - self.current_lufs
            headroom = self.max_peak - self.current_peak
            target = min(lufs_gain, headroom)
            
            # Blast protection - limit rate of change
            if self.blast_protection and self.gain_history:
                last = self.gain_history[-1]
                target = max(last - self.max_gain_change, 
                           min(last + self.max_gain_change, target))
            
            self.current_gain = self.current_gain * (1 - self.smoothing) + target * self.smoothing
            self.gain_history.append(self.current_gain)
            return self.current_gain
        return 0.0
    
    def update(self, audio_data):
        self.current_lufs = self.calculate_lufs(audio_data)
        peak = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0
        self.current_peak = (20 * np.log10(peak)) if peak > 0 else -70.0

class AudioAnalyzer:
    """Advanced spectrum analysis with imperfection detection."""
    def __init__(self):
        self.freq_bands = {
            'low': (20, 250), 'mid-low': (250, 800),
            'mid': (800, 3000), 'mid-high': (3000, 8000), 'high': (8000, 20000)
        }
        self.min_db = -80.0
        self.max_db = 0.0
        self.band_history = {band: deque(maxlen=200) for band in self.freq_bands}
        self.imperfections = {band: [] for band in self.freq_bands}
        self.eq_centers = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

    def analyze_spectrum(self, spectrum_db, freqs):
        band_analysis = {}
        for name, (low, high) in self.freq_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            vals = spectrum_db[mask]
            
            if len(vals) > 0:
                stats = {
                    'mean': np.mean(vals),
                    'peak': np.max(vals),
                    'std': np.std(vals),
                    'energy': np.sum(10**(vals/10)),
                    'centroid': np.sum(freqs[mask] * 10**(vals/10)) / (np.sum(10**(vals/10)) + 1e-10)
                }
                self.band_history[name].append(stats)
                band_analysis[name] = stats
            else:
                band_analysis[name] = {'mean': -80, 'peak': -80, 'std': 0, 'energy': 0, 'centroid': 0}
        return band_analysis

    def detect_imperfections(self, band_analysis):
        """Detect audio imperfections."""
        imperfections = {}
        for band, metrics in band_analysis.items():
            issues = []
            mean_db = metrics['mean']
            peak_db = metrics['peak']
            
            # Resonance
            if peak_db - mean_db > 6.0:
                issues.append({
                    'type': 'resonance',
                    'severity': (peak_db - mean_db)/6.0,
                    'freq': metrics['centroid'],
                    'correction': -(peak_db - mean_db) * 0.7
                })
            
            # Muddiness (Low end)
            if band in ['low', 'mid-low']:
                if metrics['mean'] > -20 and metrics['std'] < 2.0:
                    issues.append({
                        'type': 'muddiness',
                        'severity': 1.0,
                        'freq': metrics['centroid'],
                        'correction': -2.0
                    })
            
            # Harshness (High end)
            if band in ['mid-high', 'high']:
                if metrics['std'] > 8.0:
                    issues.append({
                        'type': 'harshness',
                        'severity': metrics['std']/10.0,
                        'freq': metrics['centroid'],
                        'correction': -3.0
                    })
            
            imperfections[band] = issues
        return imperfections

    def calculate_smart_eq(self, spectrum_db, freqs, current_gains):
        """Smart Auto-EQ calculation."""
        target_gains = list(current_gains)
        valid_db = spectrum_db[np.isfinite(spectrum_db)]
        if len(valid_db) == 0: return target_gains
        
        global_avg = np.mean(valid_db)
        if global_avg < -70: return target_gains

        for i, center in enumerate(self.eq_centers):
            f_min, f_max = center * 0.7, center * 1.4
            mask = (freqs >= f_min) & (freqs <= f_max)
            if np.sum(mask) == 0: continue
            
            band_avg = np.mean(spectrum_db[mask])
            
            # Target curve (flat-ish with slight bass boost)
            target_offset = 0
            if center < 100: target_offset = 2.5
            if center > 8000: target_offset = -1.5
            
            deviation = band_avg - (global_avg + target_offset)
            
            correction = 0
            if deviation > 4.0:  # Too loud
                correction = -min(deviation - 4.0, 6.0) * 0.5
            elif deviation < -6.0:  # Too quiet
                correction = min(abs(deviation) - 6.0, 6.0) * 0.4
            
            target_gains[i] = np.clip(current_gains[i] + correction, -12, 12)
        
        return target_gains
    
    def update_db_range(self, spectrum_db):
        """Update dynamic dB range."""
        valid = spectrum_db[np.isfinite(spectrum_db)]
        if len(valid) > 0:
            self.min_db = np.percentile(valid, 5)
            self.max_db = np.percentile(valid, 95)

# ==================== GUI WIDGETS ====================

class ProcessingLogWidget(Gtk.ScrolledWindow):
    """Widget displaying real-time processing log."""
    
    def __init__(self):
        super().__init__()
        self.set_size_request(400, 200)
        
        self.textview = Gtk.TextView()
        self.textview.set_editable(False)
        self.textview.set_wrap_mode(Gtk.WrapMode.WORD)
        self.textview.set_monospace(True)
        
        # Set dark theme
        self.textview.modify_bg(Gtk.StateType.NORMAL, Gdk.Color(5000, 5000, 5000))
        self.textview.modify_fg(Gtk.StateType.NORMAL, Gdk.Color(55000, 55000, 55000))
        
        self.add(self.textview)
        
        self.buffer = self.textview.get_buffer()
        
    def update_log(self, log_text):
        """Update log display."""
        self.buffer.set_text(log_text)
        
        # Auto-scroll to bottom
        mark = self.buffer.get_insert()
        self.textview.scroll_to_mark(mark, 0.0, True, 0.0, 1.0)

class LevelMeterWidget(Gtk.DrawingArea):
    """VU Meter widget."""
    
    def __init__(self):
        super().__init__()
        self.set_size_request(100, 200)
        self.lufs = -70.0
        self.peak = -70.0
        self.connect("draw", self.on_draw)
        
    def update(self, lufs, peak):
        self.lufs = lufs
        self.peak = peak
        self.queue_draw()
        
    def on_draw(self, widget, cr):
        width = widget.get_allocation().width
        height = widget.get_allocation().height
        
        # Background
        cr.set_source_rgb(0.1, 0.1, 0.1)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        
        # LUFS meter
        lufs_norm = (self.lufs + 70) / 70
        lufs_height = height * lufs_norm
        
        if self.lufs > -6:
            cr.set_source_rgb(1.0, 0.0, 0.0)
        elif self.lufs > -14:
            cr.set_source_rgb(1.0, 1.0, 0.0)
        else:
            cr.set_source_rgb(0.0, 1.0, 0.0)
        
        cr.rectangle(10, height - lufs_height, width/2 - 15, lufs_height)
        cr.fill()
        
        # Peak meter
        peak_norm = (self.peak + 70) / 70
        peak_height = height * peak_norm
        
        if self.peak > -1:
            cr.set_source_rgb(1.0, 0.0, 0.0)
        elif self.peak > -6:
            cr.set_source_rgb(1.0, 1.0, 0.0)
        else:
            cr.set_source_rgb(0.0, 0.8, 0.0)
        
        cr.rectangle(width/2 + 5, height - peak_height, width/2 - 15, peak_height)
        cr.fill()
        
        # Labels
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.move_to(5, 15)
        cr.show_text("LUFS")
        cr.move_to(width/2, 15)
        cr.show_text("Peak")

class SpectrumWidget(Gtk.DrawingArea):
    """Spectrum visualization widget."""
    
    def __init__(self):
        super().__init__()
        self.set_size_request(600, 250)
        self.spectrum_data = None
        self.connect("draw", self.on_draw)
        
    def on_draw(self, widget, cr):
        width = widget.get_allocation().width
        height = widget.get_allocation().height
        
        # Background
        cr.set_source_rgb(0.05, 0.05, 0.05)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        
        if self.spectrum_data is None:
            cr.set_source_rgb(0.5, 0.5, 0.5)
            cr.move_to(width/2 - 50, height/2)
            cr.show_text("No signal")
            return
        
        mags, freqs = self.spectrum_data
        
        # Draw spectrum
        cr.set_source_rgb(0.2, 0.8, 0.3)
        cr.set_line_width(1.5)
        
        for i in range(1, len(freqs)):
            if freqs[i] < 20 or freqs[i] > 20000:
                continue
            
            x = width * np.log10(freqs[i] / 20) / np.log10(20000 / 20)
            y = height * (1 - (mags[i] + 80) / 80)
            
            if i == 1:
                cr.move_to(x, y)
            else:
                cr.line_to(x, y)
        
        cr.stroke()

# ==================== MAIN APPLICATION ====================

class AudioAnalyzerApp(Gtk.Window):
    """Main application window."""
    
    def __init__(self):
        super().__init__(title="Audio EQ Ultimate v8.0 - Advanced RT Processing")
        self.set_default_size(1400, 900)
        
        # Core components
        self.analyzer = AudioAnalyzer()
        self.normalizer = AudioNormalizer()
        self.silence_detector = SilenceDetector(threshold_db=-50, min_duration=0.3)
        self.preset_manager = AIPresetManager()
        self.rt_log = RTProcessingLog(max_entries=200)
        
        # Load custom presets
        self.preset_manager.load_custom_presets()
        
        # State
        self.pipeline = None
        self.is_playing = False
        self.auto_eq_enabled = False
        self.normalize_enabled = False
        self.silence_bypass_enabled = True
        self.eq_gains = [0.0] * 10
        self.smoothing_factor = 0.15
        
        self.setup_ui()
        self.setup_gstreamer()
        
        # Start log update timer
        GLib.timeout_add(200, self.update_log_display)
        
        self.connect("destroy", Gtk.main_quit)

    def setup_ui(self):
        """Setup user interface."""
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_box.set_margin_top(10)
        main_box.set_margin_bottom(10)
        main_box.set_margin_start(10)
        main_box.set_margin_end(10)
        self.add(main_box)
        
        # Left panel - Visualization
        left_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        main_box.pack_start(left_vbox, True, True, 0)
        
        # Spectrum
        spec_frame = Gtk.Frame(label="Spectrum Analysis")
        self.spectrum_widget = SpectrumWidget()
        spec_frame.add(self.spectrum_widget)
        left_vbox.pack_start(spec_frame, True, True, 0)
        
        # Processing Log
        log_frame = Gtk.Frame(label="Real-Time Processing Log")
        self.log_widget = ProcessingLogWidget()
        log_frame.add(self.log_widget)
        left_vbox.pack_start(log_frame, True, True, 0)
        
        # Right panel - Controls
        right_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        main_box.pack_start(right_vbox, False, False, 0)
        
        # Source selection
        source_frame = Gtk.Frame(label="Audio Source")
        source_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        source_box.set_margin_start(5)
        source_box.set_margin_end(5)
        source_box.set_margin_top(5)
        source_box.set_margin_bottom(5)
        source_frame.add(source_box)
        right_vbox.pack_start(source_frame, False, False, 0)
        
        source_box.pack_start(Gtk.Label(label="Source:"), False, False, 0)
        self.source_combo = Gtk.ComboBoxText()
        self.source_combo.connect("changed", self.on_source_changed)
        source_box.pack_start(self.source_combo, True, True, 0)
        self.populate_sources()
        
        # Controls
        ctrl_frame = Gtk.Frame(label="Controls")
        ctrl_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        ctrl_box.set_margin_start(5)
        ctrl_box.set_margin_end(5)
        ctrl_box.set_margin_top(5)
        ctrl_box.set_margin_bottom(5)
        ctrl_frame.add(ctrl_box)
        right_vbox.pack_start(ctrl_frame, False, False, 0)
        
        self.btn_play = Gtk.Button(label="▶ Start Listening")
        self.btn_play.connect("clicked", self.on_play_pause)
        ctrl_box.pack_start(self.btn_play, False, False, 0)
        
        self.check_auto = Gtk.CheckButton(label="Smart Balance (Auto-EQ)")
        self.check_auto.connect("toggled", self.on_auto_eq_toggled)
        ctrl_box.pack_start(self.check_auto, False, False, 0)
        
        self.check_norm = Gtk.CheckButton(label="Normalization (Safe)")
        self.check_norm.connect("toggled", self.on_norm_toggled)
        ctrl_box.pack_start(self.check_norm, False, False, 0)
        
        self.check_silence = Gtk.CheckButton(label="Silence Bypass")
        self.check_silence.set_active(True)
        self.check_silence.connect("toggled", self.on_silence_toggled)
        ctrl_box.pack_start(self.check_silence, False, False, 0)
        
        btn_reset = Gtk.Button(label="Reset EQ")
        btn_reset.connect("clicked", self.on_reset_eq)
        ctrl_box.pack_start(btn_reset, False, False, 0)
        
        btn_export_log = Gtk.Button(label="Export Log")
        btn_export_log.connect("clicked", self.on_export_log)
        ctrl_box.pack_start(btn_export_log, False, False, 0)
        
        # Preset Manager
        preset_frame = Gtk.Frame(label="AI Preset Manager")
        preset_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        preset_box.set_margin_start(5)
        preset_box.set_margin_end(5)
        preset_box.set_margin_top(5)
        preset_box.set_margin_bottom(5)
        preset_frame.add(preset_box)
        right_vbox.pack_start(preset_frame, False, False, 0)
        
        preset_select_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        preset_box.pack_start(preset_select_box, False, False, 0)
        
        self.preset_combo = Gtk.ComboBoxText()
        for name in sorted(self.preset_manager.presets.keys()):
            self.preset_combo.append_text(name)
        self.preset_combo.set_active(0)
        self.preset_combo.connect("changed", self.on_preset_changed)
        preset_select_box.pack_start(self.preset_combo, True, True, 0)
        
        btn_apply_preset = Gtk.Button(label="Apply")
        btn_apply_preset.connect("clicked", self.on_apply_preset)
        preset_select_box.pack_start(btn_apply_preset, False, False, 0)
        
        self.check_auto_preset = Gtk.CheckButton(label="Auto-Select Preset (AI)")
        self.check_auto_preset.connect("toggled", self.on_auto_preset_toggled)
        preset_box.pack_start(self.check_auto_preset, False, False, 0)
        
        self.preset_info_label = Gtk.Label(label="")
        self.preset_info_label.set_line_wrap(True)
        self.preset_info_label.set_max_width_chars(30)
        preset_box.pack_start(self.preset_info_label, False, False, 0)
        
        btn_save_preset = Gtk.Button(label="Save Current as Preset")
        btn_save_preset.connect("clicked", self.on_save_preset)
        preset_box.pack_start(btn_save_preset, False, False, 0)
        
        # Level Meter
        meter_frame = Gtk.Frame(label="Levels")
        self.level_meter = LevelMeterWidget()
        meter_frame.add(self.level_meter)
        right_vbox.pack_start(meter_frame, False, False, 0)
        
        # Info Labels
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        info_box.set_margin_start(5)
        info_box.set_margin_end(5)
        right_vbox.pack_start(info_box, False, False, 0)
        
        self.lbl_lufs = Gtk.Label(label="LUFS: --")
        info_box.pack_start(self.lbl_lufs, False, False, 0)
        
        self.lbl_peak = Gtk.Label(label="Peak: --")
        info_box.pack_start(self.lbl_peak, False, False, 0)
        
        self.lbl_gain = Gtk.Label(label="Gain: 0.0 dB")
        info_box.pack_start(self.lbl_gain, False, False, 0)
        
        self.lbl_silence = Gtk.Label(label="Active")
        info_box.pack_start(self.lbl_silence, False, False, 0)
        
        # EQ Sliders
        eq_frame = Gtk.Frame(label="Equalizer (10 Bands)")
        scrolled = Gtk.ScrolledWindow()
        eq_frame.add(scrolled)
        right_vbox.pack_start(eq_frame, True, True, 0)
        
        slider_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        scrolled.add(slider_box)
        
        self.sliders = []
        bands = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        for i, b in enumerate(bands):
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            lbl = Gtk.Label(label=f"{b}Hz")
            lbl.set_size_request(60, -1)
            
            s = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -12, 12, 0.1)
            s.set_value(0)
            s.set_size_request(150, -1)
            val_txt = Gtk.Label(label="0.0")
            val_txt.set_size_request(50, -1)
            
            s.connect("value-changed", self.on_user_eq_change, i, val_txt)
            
            row.pack_start(lbl, False, False, 2)
            row.pack_start(s, True, True, 0)
            row.pack_start(val_txt, False, False, 2)
            slider_box.pack_start(row, False, False, 0)
            self.sliders.append(s)

    def populate_sources(self):
        """Populate source combo box."""
        self.source_combo.remove_all()
        for device_id, name in get_pulse_sources():
            self.source_combo.append(device_id, name)
        self.source_combo.set_active(0)

    def setup_gstreamer(self):
        """Setup GStreamer pipeline."""
        self.pipeline = Gst.Pipeline.new("eq-pipeline")
        
        self.src = Gst.ElementFactory.make("pulsesrc", "src")
        active = self.source_combo.get_active_id()
        if active: 
            self.src.set_property("device", active)
        
        caps = Gst.ElementFactory.make("capsfilter", "caps")
        caps.set_property("caps", Gst.Caps.from_string("audio/x-raw,rate=44100,channels=2,format=F32LE"))
        
        self.equalizer = Gst.ElementFactory.make("equalizer-10bands", "eq")
        self.volume = Gst.ElementFactory.make("volume", "vol")
        
        self.spectrum = Gst.ElementFactory.make("spectrum", "spec")
        self.spectrum.set_property("bands", 1024)
        self.spectrum.set_property("threshold", -90)
        self.spectrum.set_property("post-messages", True)
        self.spectrum.set_property("message-magnitude", True)
        self.spectrum.set_property("interval", 50000000)
        
        self.sink = Gst.ElementFactory.make("autoaudiosink", "sink")
        
        for el in [self.src, caps, self.equalizer, self.volume, self.spectrum, self.sink]:
            self.pipeline.add(el)
            
        self.src.link(caps)
        caps.link(self.equalizer)
        self.equalizer.link(self.volume)
        self.volume.link(self.spectrum)
        self.spectrum.link(self.sink)
        
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

    def on_source_changed(self, combo):
        """Handle source change."""
        dev = combo.get_active_id()
        if not dev: return
        state = self.pipeline.get_state(0)[1]
        self.pipeline.set_state(Gst.State.NULL)
        self.src.set_property("device", dev)
        if state == Gst.State.PLAYING:
            self.pipeline.set_state(Gst.State.PLAYING)

    def on_play_pause(self, btn):
        """Toggle playback."""
        if not self.is_playing:
            self.pipeline.set_state(Gst.State.PLAYING)
            btn.set_label("■ Stop")
            self.is_playing = True
        else:
            self.pipeline.set_state(Gst.State.PAUSED)
            btn.set_label("▶ Start Listening")
            self.is_playing = False

    def on_auto_eq_toggled(self, btn):
        """Toggle auto-EQ."""
        self.auto_eq_enabled = btn.get_active()
        self.rt_log.log_auto_eq_toggle(self.auto_eq_enabled)

    def on_norm_toggled(self, btn):
        """Toggle normalization."""
        self.normalize_enabled = btn.get_active()
        if not self.normalize_enabled:
            self.volume.set_property("volume", 1.0)
            self.lbl_gain.set_text("Gain: 0.0 dB")

    def on_silence_toggled(self, btn):
        """Toggle silence bypass."""
        self.silence_bypass_enabled = btn.get_active()

    def on_reset_eq(self, btn):
        """Reset EQ to flat."""
        for i, s in enumerate(self.sliders):
            old_val = s.get_value()
            s.set_value(0)
            if old_val != 0:
                self.rt_log.log_eq_change(i, old_val, 0, "Reset")

    def on_export_log(self, btn):
        """Export processing log."""
        filename = f"eq_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if self.rt_log.export_log(filename):
            dialog = Gtk.MessageDialog(
                transient_for=self,
                flags=0,
                message_type=Gtk.MessageType.INFO,
                buttons=Gtk.ButtonsType.OK,
                text=f"Log exported to {filename}"
            )
            dialog.run()
            dialog.destroy()

    def on_preset_changed(self, combo):
        """Update preset info display."""
        name = combo.get_active_text()
        if name:
            info = self.preset_manager.get_preset_info(name)
            desc = info.get('description', '')
            tags = ', '.join(info.get('tags', []))
            self.preset_info_label.set_text(f"{desc}\n[{tags}]")

    def on_apply_preset(self, btn):
        """Apply selected preset."""
        name = self.preset_combo.get_active_text()
        if name:
            gains = self.preset_manager.get_preset(name)
            for i, gain in enumerate(gains):
                old_val = self.sliders[i].get_value()
                self.sliders[i].set_value(gain)
                if abs(old_val - gain) > 0.1:
                    self.rt_log.log_eq_change(i, old_val, gain, f"Preset: {name}")
            self.rt_log.log_preset_change(name, auto_selected=False)

    def on_auto_preset_toggled(self, btn):
        """Toggle auto preset selection."""
        self.preset_manager.auto_select_enabled = btn.get_active()

    def on_save_preset(self, btn):
        """Save current EQ as preset."""
        dialog = Gtk.Dialog(title="Save Preset", parent=self, flags=0)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                          Gtk.STOCK_OK, Gtk.ResponseType.OK)
        
        box = dialog.get_content_area()
        box.set_spacing(10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)
        
        box.pack_start(Gtk.Label(label="Preset Name:"), False, False, 0)
        entry_name = Gtk.Entry()
        entry_name.set_text("Custom Preset")
        box.pack_start(entry_name, False, False, 0)
        
        box.pack_start(Gtk.Label(label="Description:"), False, False, 0)
        entry_desc = Gtk.Entry()
        box.pack_start(entry_desc, False, False, 0)
        
        dialog.show_all()
        response = dialog.run()
        
        if response == Gtk.ResponseType.OK:
            name = entry_name.get_text()
            desc = entry_desc.get_text()
            gains = [s.get_value() for s in self.sliders]
            self.preset_manager.save_custom_preset(name, gains, desc)
            self.preset_combo.append_text(name)
        
        dialog.destroy()

    def on_user_eq_change(self, scale, idx, label):
        """Handle manual EQ change."""
        val = scale.get_value()
        label.set_text(f"{val:.1f}")
        if not self.auto_eq_enabled:
            old_val = self.eq_gains[idx]
            self.eq_gains[idx] = val
            self.equalizer.set_property(f"band{idx}", val)
            if abs(old_val - val) > 0.1:
                self.rt_log.log_eq_change(idx, old_val, val, "Manual")

    def update_slider_ui(self, idx, val):
        """Update slider without triggering callback."""
        scale = self.sliders[idx]
        scale.handler_block_by_func(self.on_user_eq_change)
        scale.set_value(val)
        scale.handler_unblock_by_func(self.on_user_eq_change)
        parent = scale.get_parent()
        lbl = parent.get_children()[2]
        lbl.set_text(f"{val:.1f}")

    def on_message(self, bus, msg):
        """Handle GStreamer messages."""
        if msg.type == Gst.MessageType.ELEMENT:
            st = msg.get_structure()
            if st.get_name() == "spectrum":
                mags = np.array(st.get_value("magnitude"))
                rate = 44100
                freqs = np.linspace(0, rate/2, len(mags))
                
                # Update analyzer
                self.analyzer.update_db_range(mags)
                band_res = self.analyzer.analyze_spectrum(mags, freqs)
                imperfections = self.analyzer.detect_imperfections(band_res)
                
                # Log detected imperfections
                for band, issues in imperfections.items():
                    for issue in issues:
                        self.rt_log.log_imperfection_detected(
                            band, issue['type'], issue['severity'], issue['correction']
                        )
                
                # Check silence
                energy = np.mean(10**(mags/10))
                rms_val = np.sqrt(energy)
                rms_db = 20 * np.log10(rms_val + 1e-10)
                
                is_silent = self.silence_detector.update(rms_db)
                
                if self.silence_bypass_enabled and is_silent:
                    if not self.silence_detector.silence_start:
                        duration = self.silence_detector.get_silence_duration()
                        self.rt_log.log_silence_bypass(True, duration)
                    GLib.idle_add(self.lbl_silence.set_text, 
                                f"Bypassed ({self.silence_detector.get_silence_duration():.1f}s)")
                    return  # Skip processing
                else:
                    if self.silence_detector.silence_start:
                        self.rt_log.log_silence_bypass(False)
                    GLib.idle_add(self.lbl_silence.set_text, "Active")
                
                # Auto-preset selection
                if self.preset_manager.auto_select_enabled:
                    auto_preset = self.preset_manager.auto_select_preset(band_res, imperfections)
                    if auto_preset and auto_preset != self.preset_manager.current_preset:
                        gains = self.preset_manager.get_preset(auto_preset)
                        for i, gain in enumerate(gains):
                            GLib.idle_add(self.update_slider_ui, i, gain)
                            self.eq_gains[i] = gain
                            self.equalizer.set_property(f"band{i}", gain)
                        self.rt_log.log_preset_change(auto_preset, auto_selected=True)
                        self.preset_manager.current_preset = auto_preset
                        GLib.idle_add(self.preset_combo.set_active_id, auto_preset)
                
                # Auto-EQ
                if self.auto_eq_enabled:
                    target_gains = self.analyzer.calculate_smart_eq(mags, freqs, self.eq_gains)
                    for i in range(10):
                        curr = self.eq_gains[i]
                        tgt = target_gains[i]
                        new_val = curr * (1 - self.smoothing_factor) + tgt * self.smoothing_factor
                        
                        if abs(new_val - curr) > 0.1:
                            self.rt_log.log_eq_change(i, curr, new_val, "Auto-EQ")
                        
                        self.eq_gains[i] = new_val
                        self.equalizer.set_property(f"band{i}", new_val)
                        GLib.idle_add(self.update_slider_ui, i, new_val)
                
                # Normalization
                if self.normalize_enabled or True:
                    self.normalizer.update(np.array([rms_val]))
                    
                    GLib.idle_add(self.lbl_lufs.set_text, f"LUFS: {self.normalizer.current_lufs:.1f}")
                    GLib.idle_add(self.lbl_peak.set_text, f"Peak: {self.normalizer.current_peak:.1f}")
                    GLib.idle_add(self.level_meter.update, 
                                self.normalizer.current_lufs, self.normalizer.current_peak)

                    if self.normalize_enabled:
                        gain_db = self.normalizer.calculate_gain()
                        gain_lin = float(10**(gain_db/20))
                        gain_lin = max(0.0, min(gain_lin, 4.0))
                        
                        self.volume.set_property("volume", gain_lin)
                        GLib.idle_add(self.lbl_gain.set_text, f"Gain: {gain_db:+.1f} dB")
                        
                        self.rt_log.log_normalization(gain_db, 
                                                     self.normalizer.current_lufs,
                                                     self.normalizer.current_peak)

                # Update visualization
                GLib.idle_add(self.spectrum_widget.queue_draw)
                self.spectrum_widget.spectrum_data = (mags, freqs)

    def update_log_display(self):
        """Update log display widget."""
        log_text = self.rt_log.get_formatted_log(count=30)
        self.log_widget.update_log(log_text)
        return True  # Continue timer

def main():
    create_virtual_sink()
    app = AudioAnalyzerApp()
    app.show_all()
    Gtk.main()

if __name__ == "__main__":
    main()
