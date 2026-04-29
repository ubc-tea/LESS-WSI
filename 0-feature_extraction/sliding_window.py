"""
Sliding Window Patch Extraction from Whole Slide Images (WSI)

This script extracts patches from .svs WSI files using a sliding window approach.
It filters patches based on mean intensity and standard deviation thresholds.

Dependencies:
- openslide-python
- numpy
- matplotlib
- pandas
- opencv-python
"""

import os
import glob
import argparse
import numpy as np
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from matplotlib import pyplot as plt


def create_patches(args):
    """
    Extract patches from WSI files in the specified directory.

    Args:
        args: Parsed command-line arguments containing wsi_path.
    """
    # Get list of .svs files
    filelist = glob.glob(os.path.join(args.wsi_path, 'images', '*.svs'))
    total_files = len(filelist)

    for i, item in enumerate(filelist):
        print(f"Processing file {i+1}/{total_files}: {item}")

        if not item.endswith('.svs'):
            continue

        # Open the slide
        slide = open_slide(item)
        slide_props = slide.properties

        # Define save directories (per-tile-size so 128 and 256 don't collide)
        slide_name = os.path.splitext(os.path.basename(item))[0]
        save_dir = os.path.join(
            args.wsi_path, f'save_patches_{args.tile_size}', slide_name
        )
        discard_dir = os.path.join(
            args.wsi_path, f'discard_patches_{args.tile_size}', slide_name
        )

        if os.path.exists(save_dir):
            print(f"Skipping {slide_name} as patches already exist.")
            continue

        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(discard_dir, exist_ok=True)

        # Print slide information
        level_10x = slide.level_count - 1
        print(f"Level 10x: {level_10x}")
        print(f"Vendor: {slide_props.get('openslide.vendor', 'Unknown')}")
        print(f"Pixel size X (um): {slide_props.get('openslide.mpp-x', 'Unknown')}")
        print(f"Pixel size Y (um): {slide_props.get('openslide.mpp-y', 'Unknown')}")

        dims = slide.level_dimensions
        num_levels = len(dims)
        print(f"Number of levels: {num_levels}")
        print(f"Level dimensions: {dims}")

        factors = slide.level_downsamples
        print(f"Level downsampling factors: {factors}")

        # Create DeepZoom generator at the requested tile size
        tiles = DeepZoomGenerator(
            slide, tile_size=args.tile_size, overlap=0, limit_bounds=False
        )
        # Pick the DeepZoom level whose downsample best matches the requested
        # magnification. DeepZoom's last level corresponds to the slide's
        # native resolution; the paper extracts at 10x for cytology cohorts.
        # Original code used level_tiles[-3] for cols/rows but a literal
        # level=15 for get_tile, which only matched on cohorts whose level
        # count was exactly 18. We now resolve a single level index from
        # --dz_level_offset (default -3, matching the original behaviour for
        # 18-level slides) and use it consistently.
        dz_level = len(tiles.level_tiles) + args.dz_level_offset
        if not 0 <= dz_level < len(tiles.level_tiles):
            raise RuntimeError(
                f"DeepZoom level {dz_level} out of range "
                f"[0, {len(tiles.level_tiles)}) for slide {slide_name}; "
                f"adjust --dz_level_offset"
            )
        cols, rows = tiles.level_tiles[dz_level]
        print(f'DeepZoom level used: {dz_level} (offset={args.dz_level_offset})')
        print(f'Total columns: {cols}, Total rows: {rows}')

        count_save = 0
        count_discard = 0

        # Center crop is half the tile size on each side (e.g. 128x128 inside
        # a 256-px tile, 64x64 inside a 128-px tile). This preserves the
        # ratio used by the paper's screening rule across scales.
        half_center = args.tile_size // 4

        # Process each tile
        for row in range(rows):
            for col in range(cols):
                tile_name = os.path.join(save_dir, f'{col}_{row}')
                discard_tile_name = os.path.join(discard_dir, f'{col}_{row}')

                temp_tile = tiles.get_tile(dz_level, (col, row))
                temp_tile_rgb = temp_tile.convert('RGB')
                temp_tile_np = np.array(temp_tile_rgb)

                # Drop edge tiles that are not the requested size.
                if (temp_tile_np.shape[0] != args.tile_size
                        or temp_tile_np.shape[1] != args.tile_size):
                    continue

                # Calculate thresholds
                mean_intensity = temp_tile_np.mean()
                std_intensity = temp_tile_np.std()
                mid_point = temp_tile_np.shape[0] // 2
                center_mean = temp_tile_np[
                    mid_point - half_center:mid_point + half_center,
                    mid_point - half_center:mid_point + half_center,
                ].mean()

                # Apply filters: mean < mean_thr, std > std_thr,
                # center mean < center_mean_thr.
                if (mean_intensity < args.mean_thr
                        and std_intensity > args.std_thr
                        and center_mean < args.center_mean_thr):
                    plt.imsave(f"{tile_name}_m.png", temp_tile_np)
                    count_save += 1
                else:
                    count_discard += 1

        print(f'Saved tiles: {count_save}')
        print(f'Discarded tiles: {count_discard}')
        print('-' * 50)


def main():
    parser = argparse.ArgumentParser(description='Extract patches from WSI files')
    parser.add_argument('--wsi_path', type=str, default='/bigdata/urine',
                        help='Path to the directory containing WSI files')
    parser.add_argument('--tile_size', type=int, default=256, choices=[128, 256],
                        help='Tile size in pixels. Run with 256 and 128 to '
                             'produce the two scales the paper uses. The '
                             'screening center crop scales with this '
                             '(half_center = tile_size // 4).')
    parser.add_argument('--dz_level_offset', type=int, default=-3,
                        help='DeepZoom level used for tile extraction, given '
                             'as a negative offset from the highest level '
                             '(default -3 matches the paper for 18-level '
                             'slides). The same level is now used for both '
                             'level_tiles[] and get_tile().')
    parser.add_argument('--mean_thr', type=float, default=236.0,
                        help='Tile is kept only if its mean intensity is '
                             'below this (default 236, paper).')
    parser.add_argument('--std_thr', type=float, default=10.0,
                        help='Tile is kept only if its std is above this '
                             '(default 10, paper).')
    parser.add_argument('--center_mean_thr', type=float, default=236.0,
                        help='Tile is kept only if its center crop mean '
                             'intensity is below this (default 236, paper).')

    args = parser.parse_args()
    create_patches(args)


if __name__ == '__main__':
    main()