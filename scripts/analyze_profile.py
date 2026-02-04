#!/usr/bin/env python3
"""
Analyze PyTorch profiler trace files to get operation statistics.

Usage:
    python scripts/analyze_profile.py /path/to/trace.json.gz --stream 20 --top-k 40 --output analysis_results.txt
"""

import gzip
import json
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class TeeOutput:
    """Write to both stdout and a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


def load_trace(trace_path: str) -> dict:
    """Load a gzipped or plain JSON trace file."""
    path = Path(trace_path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def analyze_trace(trace_data: dict, top_k: int = 30, filter_pattern: str = None, 
                   time_range: tuple = None, stream_id: int = None, print_results: bool = True):
    """
    Analyze trace events and aggregate by operation name.
    
    Args:
        trace_data: Loaded trace JSON
        top_k: Number of top operations to show
        filter_pattern: Optional filter pattern for operation names
        time_range: Optional (start_us, end_us) to filter events
        stream_id: Optional stream ID to filter
        print_results: Whether to print results
    """
    events = trace_data.get('traceEvents', [])
    
    # Aggregate by operation name
    op_stats = defaultdict(lambda: {'count': 0, 'total_us': 0, 'durations': []})
    
    # Track events by category
    category_stats = defaultdict(lambda: {'count': 0, 'total_us': 0})
    
    for event in events:
        # Only process complete events (ph='X') or duration events
        ph = event.get('ph', '')
        if ph not in ['X', 'i']:  # X=complete event, i=instant
            continue
            
        name = event.get('name', 'unknown')
        dur = event.get('dur', 0)  # duration in microseconds
        cat = event.get('cat', 'unknown')
        ts = event.get('ts', 0)
        
        if dur <= 0:
            continue
        
        # Filter by time range
        if time_range:
            if ts < time_range[0] or ts > time_range[1]:
                continue
        
        # Filter by stream
        if stream_id is not None:
            tid = event.get('tid')
            event_stream = event.get('args', {}).get('stream')
            if tid != stream_id and event_stream != stream_id:
                continue
            
        # Apply filter if specified
        if filter_pattern and filter_pattern.lower() not in name.lower():
            continue
        
        op_stats[name]['count'] += 1
        op_stats[name]['total_us'] += dur
        op_stats[name]['durations'].append(dur)
        
        category_stats[cat]['count'] += 1
        category_stats[cat]['total_us'] += dur
    
    # Sort by total time
    sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]['total_us'], reverse=True)
    total_time = sum(op['total_us'] for op in op_stats.values())
    sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]['total_us'], reverse=True)
    
    if print_results:
        # Print results
        print("\n" + "="*100)
        print(f"TOP {top_k} OPERATIONS BY TOTAL TIME")
        print("="*100)
        print(f"{'Operation':<60} {'Count':>8} {'Total (s)':>12} {'Avg (s)':>10} {'%':>6}")
        print("-"*100)
        
        for name, stats in sorted_ops[:top_k]:
            count = stats['count']
            total_s = stats['total_us'] / 1000000  # Convert to seconds
            avg_s = total_s / count if count > 0 else 0
            pct = (stats['total_us'] / total_time * 100) if total_time > 0 else 0
            
            # Truncate long names
            display_name = name[:57] + '...' if len(name) > 60 else name
            print(f"{display_name:<60} {count:>8} {total_s:>12.1f} {avg_s:>10.3f} {pct:>5.1f}%")
        
        # Print category summary
        print("\n" + "="*100)
        print("CATEGORY SUMMARY")
        print("="*100)
        print(f"{'Category':<40} {'Count':>10} {'Total (s)':>15} {'%':>8}")
        print("-"*100)
        
        for cat, stats in sorted_cats[:15]:
            total_s = stats['total_us'] / 1000000  # Convert to seconds
            pct = (stats['total_us'] / total_time * 100) if total_time > 0 else 0
            print(f"{cat:<40} {stats['count']:>10} {total_s:>15.1f} {pct:>7.1f}%")
    
    return op_stats, category_stats, total_time


def find_flash_attn_ops(trace_data: dict, time_range: tuple = None, stream_id: int = None, print_results: bool = True):
    """Find and summarize flash attention operations."""
    events = trace_data.get('traceEvents', [])
    
    flash_attn_events = []
    for event in events:
        name = event.get('name', '')
        dur = event.get('dur', 0)
        ts = event.get('ts', 0)
        
        if 'flash' in name.lower() or 'attn' in name.lower():
            if dur > 0:
                # Filter by time range
                if time_range:
                    if ts < time_range[0] or ts > time_range[1]:
                        continue
                
                # Filter by stream
                if stream_id is not None:
                    tid = event.get('tid')
                    event_stream = event.get('args', {}).get('stream')
                    if tid != stream_id and event_stream != stream_id:
                        continue
                
                flash_attn_events.append({
                    'name': name,
                    'dur_ms': dur / 1000,
                    'ts': ts
                })
    
    # Group by name
    attn_by_name = defaultdict(list)
    for e in flash_attn_events:
        attn_by_name[e['name']].append(e['dur_ms'])
    
    total_flash_time = sum(sum(d) for d in attn_by_name.values())
    
    if print_results and flash_attn_events:
        print("\n" + "="*100)
        print("FLASH ATTENTION OPERATIONS")
        print("="*100)
        
        for name, durations in sorted(attn_by_name.items(), key=lambda x: sum(x[1]), reverse=True):
            total_s = sum(durations) / 1000  # Convert to seconds
            avg_s = total_s / len(durations)
            print(f"{name[:70]:<70}")
            print(f"  Count: {len(durations)}, Total: {total_s:.1f}s, Avg: {avg_s:.3f}s")
    
    return flash_attn_events, attn_by_name, total_flash_time


def find_kernel_ops(trace_data: dict, kernel_patterns: list = None, time_range: tuple = None, 
                    stream_id: int = None, print_results: bool = True):
    """Find specific kernel operations."""
    if kernel_patterns is None:
        kernel_patterns = ['flash', 'gemm', 'conv', 'norm', 'softmax', 'elementwise']
    
    events = trace_data.get('traceEvents', [])
    
    kernel_stats = defaultdict(lambda: {'count': 0, 'total_ms': 0})
    
    for event in events:
        name = event.get('name', '').lower()
        dur = event.get('dur', 0)
        cat = event.get('cat', '')
        ts = event.get('ts', 0)
        
        if dur <= 0:
            continue
        
        # Filter by time range
        if time_range:
            if ts < time_range[0] or ts > time_range[1]:
                continue
        
        # Filter by stream
        if stream_id is not None:
            tid = event.get('tid')
            event_stream = event.get('args', {}).get('stream')
            if tid != stream_id and event_stream != stream_id:
                continue
        
        # Check if this is a CUDA kernel
        if 'kernel' not in cat.lower() and 'cuda' not in cat.lower():
            continue
            
        for pattern in kernel_patterns:
            if pattern.lower() in name:
                kernel_stats[pattern]['count'] += 1
                kernel_stats[pattern]['total_ms'] += dur / 1000
                break
    
    if print_results and kernel_stats:
        print("\n" + "="*100)
        print("KERNEL TYPE SUMMARY")
        print("="*100)
        print(f"{'Kernel Type':<20} {'Count':>10} {'Total (s)':>15}")
        print("-"*50)
        
        for pattern, stats in sorted(kernel_stats.items(), key=lambda x: x[1]['total_ms'], reverse=True):
            total_s = stats['total_ms'] / 1000  # Convert to seconds
            print(f"{pattern:<20} {stats['count']:>10} {total_s:>15.1f}")
    
    return kernel_stats


def print_stream_samples(trace_data: dict, stream_id: int, num_samples: int = 20):
    """Print first N events from a specific stream."""
    events = trace_data.get('traceEvents', [])
    
    # Categories that are user annotations (span entire regions)
    annotation_categories = {'gpu_user_annotation', 'user_annotation', 'python_function'}
    
    # Filter events for this stream
    stream_events = []
    kernel_events = []
    for event in events:
        tid = event.get('tid')
        args = event.get('args', {})
        event_stream = args.get('stream')
        
        if tid == stream_id or event_stream == stream_id:
            stream_events.append(event)
            # Also track kernel events (non-annotation)
            cat = event.get('cat', '')
            name = event.get('name', '')
            if cat not in annotation_categories and not name.startswith('fastvideo.region::'):
                kernel_events.append(event)
    
    # Sort by timestamp
    kernel_events.sort(key=lambda x: x.get('ts', 0))
    
    print(f"\n" + "="*130)
    print(f"FIRST {num_samples} KERNEL EVENTS ON STREAM {stream_id} (EXCLUDING ANNOTATIONS)")
    print("="*130)
    print(f"{'Timestamp':>15} {'Duration':>12} {'Name':<60} {'Category':<25}")
    print("-"*130)
    
    for event in kernel_events[:num_samples]:
        ts = event.get('ts', 0)
        dur = event.get('dur', 0)
        name = event.get('name', '')[:60]
        cat = event.get('cat', '')[:25]
        
        ts_sec = ts / 1e6
        dur_ms = dur / 1000
        
        print(f"{ts_sec:>12.6f}s {dur_ms:>10.3f}ms {name:<60} {cat:<25}")
    
    print(f"\nTotal kernel events on stream {stream_id}: {len(kernel_events)}")
    
    return stream_events


def list_streams(trace_data: dict):
    """List all available streams/threads in the trace."""
    events = trace_data.get('traceEvents', [])
    
    streams = defaultdict(lambda: {'count': 0, 'total_dur_ms': 0})
    
    for event in events:
        tid = event.get('tid')
        dur = event.get('dur', 0)
        
        if tid is not None:
            streams[tid]['count'] += 1
            streams[tid]['total_dur_ms'] += dur / 1000
    
    print("\n" + "="*80)
    print("AVAILABLE STREAMS/THREADS")
    print("="*80)
    print(f"{'Stream/TID':<15} {'Event Count':>12} {'Total (s)':>15}")
    print("-"*80)
    
    for tid, stats in sorted(streams.items(), key=lambda x: x[1]['total_dur_ms'], reverse=True):
        total_s = stats['total_dur_ms'] / 1000  # Convert to seconds
        print(f"{tid:<15} {stats['count']:>12} {total_s:>12.1f}")
    
    return streams


def detect_chunks(trace_data: dict, gap_threshold_ms: float = 100.0, stream_id: int = None):
    """
    Detect chunk boundaries based on gaps in GPU activity.
    
    Args:
        trace_data: Loaded trace JSON
        gap_threshold_ms: Minimum gap (in ms) to consider as chunk boundary
        stream_id: If provided, only consider events from this stream/thread
        
    Returns:
        List of (chunk_start_us, chunk_end_us) tuples
    """
    events = trace_data.get('traceEvents', [])
    
    # Categories to skip (user annotations that span entire regions)
    skip_categories = {'gpu_user_annotation', 'user_annotation', 'python_function'}
    skip_name_prefixes = ('fastvideo.region::', 'ProfilerStep', 'cudaLaunchKernel')
    
    # Get all events with timing info, sorted by start time
    timed_events = []
    for event in events:
        ts = event.get('ts', 0)
        dur = event.get('dur', 0)
        cat = event.get('cat', '')
        name = event.get('name', '')
        
        if ts <= 0 or dur <= 0:
            continue
        
        # Skip user annotation regions - they span entire chunks
        if cat in skip_categories:
            continue
        if name.startswith(skip_name_prefixes):
            continue
        
        # Filter by stream if specified
        if stream_id is not None:
            # Check thread ID (tid)
            tid = event.get('tid')
            # Also check args.stream
            args = event.get('args', {})
            event_stream = args.get('stream')
            
            # Match either tid or stream
            if tid != stream_id and event_stream != stream_id:
                continue
        
        timed_events.append({
            'ts': ts,
            'end': ts + dur,
            'name': name,
            'cat': cat,
            'tid': event.get('tid'),
            'stream': event.get('args', {}).get('stream')
        })
    
    if not timed_events:
        print(f"No kernel events found for stream {stream_id} (after filtering user annotations)")
        return []
    
    print(f"Found {len(timed_events)} kernel events for stream {stream_id} (excluding user annotations)")
    
    # Sort by start time
    timed_events.sort(key=lambda x: x['ts'])
    
    # Find gaps
    chunks = []
    chunk_start = timed_events[0]['ts']
    last_end = timed_events[0]['end']
    
    gap_threshold_us = gap_threshold_ms * 1000
    
    for event in timed_events[1:]:
        gap = event['ts'] - last_end
        if gap > gap_threshold_us:
            # Found a chunk boundary
            chunks.append((chunk_start, last_end))
            chunk_start = event['ts']
        last_end = max(last_end, event['end'])
    
    # Add last chunk
    chunks.append((chunk_start, last_end))
    
    return chunks


def analyze_per_chunk(trace_data: dict, chunks: list, top_k: int = 40, plot_file: str = None):
    """
    Analyze ALL operations per chunk using the full analysis functions.
    
    Note: stream_id is only used for chunk boundary detection.
    Per-chunk analysis includes ALL operations (all streams) within the time range.
    
    Args:
        trace_data: Loaded trace JSON
        chunks: List of (start_us, end_us) chunk boundaries
        top_k: Number of top operations to show per chunk
        plot_file: If provided, save kernel type bar plot to this file
    """
    
    # Collect kernel stats and category stats for all chunks
    all_chunk_kernel_stats = []
    all_chunk_category_stats = []
    
    for i, (start, end) in enumerate(chunks):
        duration_ms = (end - start) / 1000
        duration_s = duration_ms / 1000
        
        print("\n" + "#"*120)
        print(f"#  CHUNK {i}: {start/1e6:.3f}s - {end/1e6:.3f}s (duration: {duration_s:.1f}s)")
        print("#"*120)
        
        time_range = (start, end)
        
        # Top operations (ALL streams) - also get category stats
        _, category_stats, _ = analyze_trace(trace_data, top_k=top_k, time_range=time_range, stream_id=None)
        
        # Flash attention (ALL streams)
        find_flash_attn_ops(trace_data, time_range=time_range, stream_id=None)
        
        # Kernel summary (ALL streams)
        kernel_stats = find_kernel_ops(trace_data, time_range=time_range, stream_id=None)
        all_chunk_kernel_stats.append({
            'chunk_id': i,
            'duration_ms': duration_ms,
            'kernel_stats': dict(kernel_stats)
        })
        all_chunk_category_stats.append({
            'chunk_id': i,
            'duration_ms': duration_ms,
            'category_stats': dict(category_stats)
        })
    
    # Print comparison table
    print("\n" + "="*120)
    print("CHUNK COMPARISON SUMMARY")
    print("="*120)
    print(f"{'Chunk':<10} {'Duration (s)':>12} {'Flash Attn (s)':>15} {'Total Ops (s)':>15}")
    print("-"*60)
    
    for i, (start, end) in enumerate(chunks):
        duration_s = (end - start) / 1000000  # Convert to seconds
        time_range = (start, end)
        
        # Get flash attn time (ALL streams)
        _, _, flash_time = find_flash_attn_ops(trace_data, time_range=time_range, stream_id=None, print_results=False)
        flash_time_s = flash_time / 1000  # Convert to seconds
        
        # Get total time (ALL streams)
        _, _, total_time = analyze_trace(trace_data, time_range=time_range, stream_id=None, print_results=False)
        total_time_s = total_time / 1000000 if total_time else 0  # Convert to seconds
        
        print(f"Chunk {i:<5} {duration_s:>10.1f} {flash_time_s:>13.1f} {total_time_s:>13.1f}")
    
    # Print kernel type comparison table
    print("\n" + "="*120)
    print("KERNEL TYPE PER CHUNK (seconds)")
    print("="*120)
    
    # Collect all kernel types
    all_kernel_types = set()
    for chunk_data in all_chunk_kernel_stats:
        all_kernel_types.update(chunk_data['kernel_stats'].keys())
    all_kernel_types = sorted(all_kernel_types)
    
    # Header
    header = f"{'Chunk':<10}"
    for kt in all_kernel_types:
        header += f" {kt:>12}"
    print(header)
    print("-"*120)
    
    # Per-chunk rows
    for chunk_data in all_chunk_kernel_stats:
        row = f"Chunk {chunk_data['chunk_id']:<5}"
        for kt in all_kernel_types:
            stats = chunk_data['kernel_stats'].get(kt, {'total_ms': 0})
            total_s = stats['total_ms'] / 1000  # Convert to seconds
            row += f" {total_s:>11.1f}s"
        print(row)
    
    # Print category comparison table
    print("\n" + "="*120)
    print("CATEGORY PER CHUNK (seconds)")
    print("="*120)
    
    # Collect all categories
    all_categories = set()
    for chunk_data in all_chunk_category_stats:
        all_categories.update(chunk_data['category_stats'].keys())
    # Sort by total time across all chunks
    cat_totals = {}
    for cat in all_categories:
        cat_totals[cat] = sum(
            chunk_data['category_stats'].get(cat, {}).get('total_us', 0) 
            for chunk_data in all_chunk_category_stats
        )
    all_categories = sorted(all_categories, key=lambda x: cat_totals[x], reverse=True)[:8]  # Top 8 categories
    
    # Header
    header = f"{'Chunk':<10}"
    for cat in all_categories:
        header += f" {cat[:12]:>12}"
    print(header)
    print("-"*120)
    
    # Per-chunk rows
    for chunk_data in all_chunk_category_stats:
        row = f"Chunk {chunk_data['chunk_id']:<5}"
        for cat in all_categories:
            stats = chunk_data['category_stats'].get(cat, {'total_us': 0})
            total_s = stats['total_us'] / 1000000  # Convert to seconds
            row += f" {total_s:>11.1f}s"
        print(row)
    
    # Generate bar plots
    if plot_file and HAS_MATPLOTLIB:
        generate_kernel_bar_plot(all_chunk_kernel_stats, all_kernel_types, plot_file)
        
        # Generate category plot
        category_plot_file = plot_file.replace('.png', '_category.png')
        generate_category_bar_plot(all_chunk_category_stats, all_categories, category_plot_file)
    elif plot_file and not HAS_MATPLOTLIB:
        print(f"\nWarning: matplotlib not installed, cannot generate plot")
    
    return all_chunk_kernel_stats


def generate_kernel_bar_plot(chunk_stats: list, kernel_types: list, output_file: str):
    """Generate a grouped bar plot of kernel times per chunk."""
    
    num_chunks = len(chunk_stats)
    num_types = len(kernel_types)
    
    # Prepare data (convert ms to seconds)
    data = np.zeros((num_chunks, num_types))
    for i, chunk_data in enumerate(chunk_stats):
        for j, kt in enumerate(kernel_types):
            stats = chunk_data['kernel_stats'].get(kt, {'total_ms': 0})
            data[i, j] = stats['total_ms'] / 1000  # Convert to seconds
    
    # Create figure with larger font
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Bar positions
    x = np.arange(num_chunks)
    width = 0.8 / num_types
    
    # Colors
    colors = plt.cm.Set2(np.linspace(0, 1, num_types))
    
    # Plot bars
    for j, kt in enumerate(kernel_types):
        offset = (j - num_types/2 + 0.5) * width
        bars = ax.bar(x + offset, data[:, j], width, label=kt, color=colors[j])
        
        # Add value labels on bars
        for bar, val in zip(bars, data[:, j]):
            if val > 0:
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Chunk', fontsize=16)
    ax.set_ylabel('Time (seconds)', fontsize=16)
    ax.set_title('Kernel Type Time per Chunk', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Chunk {i}' for i in range(num_chunks)], fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nKernel bar plot saved to: {output_file}")


def generate_category_bar_plot(chunk_stats: list, categories: list, output_file: str):
    """Generate a grouped bar plot of category times per chunk."""
    
    if not HAS_MATPLOTLIB:
        return
    
    num_chunks = len(chunk_stats)
    num_cats = len(categories)
    
    # Prepare data (convert us to seconds)
    data = np.zeros((num_chunks, num_cats))
    for i, chunk_data in enumerate(chunk_stats):
        for j, cat in enumerate(categories):
            stats = chunk_data['category_stats'].get(cat, {'total_us': 0})
            data[i, j] = stats['total_us'] / 1000000  # Convert to seconds
    
    # Create figure with larger font
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Bar positions
    x = np.arange(num_chunks)
    width = 0.8 / num_cats
    
    # Colors
    colors = plt.cm.Set3(np.linspace(0, 1, num_cats))
    
    # Plot bars
    for j, cat in enumerate(categories):
        offset = (j - num_cats/2 + 0.5) * width
        bars = ax.bar(x + offset, data[:, j], width, label=cat[:15], color=colors[j])
        
        # Add value labels on bars
        for bar, val in zip(bars, data[:, j]):
            if val > 0.1:  # Only show if > 0.1s
                ax.annotate(f'{val:.1f}s',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, rotation=45)
    
    # Labels and title
    ax.set_xlabel('Chunk', fontsize=16)
    ax.set_ylabel('Time (seconds)', fontsize=16)
    ax.set_title('Category Time per Chunk', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Chunk {i}' for i in range(num_chunks)], fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Category bar plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze PyTorch profiler traces')
    parser.add_argument('trace_file', help='Path to trace file (.json or .json.gz)')
    parser.add_argument('--top-k', type=int, default=40, help='Number of top operations to show')
    parser.add_argument('--filter', type=str, default=None, help='Filter operations by pattern')
    parser.add_argument('--gap-threshold', type=float, default=300.0, 
                        help='Gap threshold (ms) for chunk detection')
    parser.add_argument('--stream', type=int, default=20, 
                        help='Stream/thread ID to use for chunk detection (default: 20)')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of sample events to print from stream (default: 20)')
    parser.add_argument('--no-chunks', action='store_true', help='Skip chunk analysis')
    parser.add_argument('--output', '-o', type=str, default='analysis_results.txt',
                        help='Output file to save results (also prints to terminal)')
    parser.add_argument('--plot', '-p', type=str, default='kernel_plot.png',
                        help='Output file for kernel type bar plot (e.g., kernel_plot.png)')
    args = parser.parse_args()
    
    # Set up tee output if requested
    tee = None
    if args.output:
        tee = TeeOutput(args.output)
        sys.stdout = tee
        print(f"Saving output to: {args.output}")
    
    print(f"Loading trace from: {args.trace_file}")
    trace_data = load_trace(args.trace_file)
    
    print(f"Loaded {len(trace_data.get('traceEvents', []))} events")
    
    # Main analysis
    analyze_trace(trace_data, top_k=args.top_k, filter_pattern=args.filter)
    
    # Flash attention specific
    find_flash_attn_ops(trace_data)
    
    # Kernel summary
    find_kernel_ops(trace_data)
    
    # List available streams
    list_streams(trace_data)
    
    # Print sample events from selected stream
    print_stream_samples(trace_data, args.stream, args.samples)
    
    # Per-chunk analysis
    if not args.no_chunks:
        print(f"\nDetecting chunks with gap threshold: {args.gap_threshold}ms on stream {args.stream}")
        chunks = detect_chunks(trace_data, gap_threshold_ms=args.gap_threshold, stream_id=args.stream)
        print(f"Detected {len(chunks)} chunks")
        
        if len(chunks) >= 1:
            # Print chunk boundaries
            print("\nChunk boundaries:")
            for i, (start, end) in enumerate(chunks):
                duration = (end - start) / 1000
                print(f"  Chunk {i}: {start/1e6:.3f}s - {end/1e6:.3f}s (duration: {duration:.1f}ms)")
            
            analyze_per_chunk(trace_data, chunks, top_k=args.top_k, plot_file=args.plot)
    
    # Close tee output
    if tee:
        sys.stdout = tee.terminal
        tee.close()
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
