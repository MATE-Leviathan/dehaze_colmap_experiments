import cv2
import math
import numpy as np
import sys
import argparse

def apply_mask(matrix, mask, fill_value):
    #print("MATRIX=", matrix)
    #print("mask=\n" ,mask)
    #print("fill value=\n", fill_value)
                 
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    print('MASKED=',masked)
    return masked.filled()

def apply_threshold(matrix, low_value=255, high_value=255):
    low_mask = matrix < low_value
    print("low mask=",low_mask)
    
    matrix = apply_mask(matrix, low_mask, low_value)
    print('Low MASK->',low_mask,'\nMatrix->',matrix)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100
    print("shape of image = ", img.shape[2])

    half_percent = percent / 200.0
    print('HALF PERCENT->',half_percent)

    channels = cv2.split(img)
    print('Channels->\n',channels)
    print('Shape->',channels[0].shape)
    print('Shape of channels->',len(channels[2]))

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2

	# find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        print('vec=',vec_size,'\nFlat=',flat)
        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]
        print("Number of columns = ", n_cols)

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        print("Lowval: ", low_val)
        print("Highval: ", high_val)
        print(flat[60])
        print(flat[11940])
        

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

def save_frame_at_time(cap, time_seconds, output_filename):
    """
    Save a single frame at the specified time to a file.

    Args:
        cap: OpenCV VideoCapture object
        time_seconds: Time in seconds where to extract the frame
        output_filename: Path to save the frame image
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(time_seconds * fps)

    print(f"Extracting frame at {time_seconds}s (frame #{frame_number})")

    # Set video to the specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read frame at {time_seconds}s")
        return False

    # Save the frame
    cv2.imwrite(output_filename, frame)
    print(f"Frame saved to '{output_filename}'")

    return True

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Dehaze video using Simplest Color Balance algorithm')
    parser.add_argument('--video', type=str, default='../GX010048.MP4',
                        help='Path to input video file')
    parser.add_argument('--mode', type=str, choices=['process', 'extract'], default='process',
                        help='Mode: "process" to dehaze video segment, "extract" to save single frame')
    parser.add_argument('--start', type=float, default=610.0,
                        help='Start time in seconds (default: 610 = 10:10)')
    parser.add_argument('--end', type=float, default=630.0,
                        help='End time in seconds (default: 630 = 10:30)')
    parser.add_argument('--extract-time', type=float, default=615.0,
                        help='Time in seconds to extract frame (default: 615 = 10:15)')
    parser.add_argument('--output-frame', type=str, default='frame_at_10m15s.png',
                        help='Output filename for extracted frame')
    parser.add_argument('--output-before', type=str, default='dehaze_output_before.mp4',
                        help='Output filename for original video')
    parser.add_argument('--output-after', type=str, default='dehaze_output_after.mp4',
                        help='Output filename for dehazed video')

    args = parser.parse_args()

    # Open video file
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.video}'")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")

    if args.mode == 'extract':
        # Extract single frame mode
        save_frame_at_time(cap, args.extract_time, args.output_frame)
        cap.release()
        print("\nExiting without processing the rest of the video.")
        sys.exit(0)

    else:
        # Process video segment mode
        start_time = args.start
        end_time = args.end

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        print(f"\nProcessing from {start_time}s to {end_time}s")
        print(f"Frame range: {start_frame} to {end_frame}")

        # Set video to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Set up video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_before = cv2.VideoWriter(args.output_before, fourcc, fps, (width, height))
        out_after = cv2.VideoWriter(args.output_after, fourcc, fps, (width, height))

        current_frame = start_frame
        processed_frames = 0

        while current_frame <= end_frame:
            ret, frame = cap.read()

            if not ret:
                print(f"Warning: Could not read frame {current_frame}")
                break

            # Save original frame
            out_before.write(frame)

            # Apply dehazing
            dehazed = simplest_cb(frame, 1)

            # Save dehazed frame
            out_after.write(dehazed)

            processed_frames += 1
            current_frame += 1

            # Print progress
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames} frames...")

        print(f"\nProcessing complete! {processed_frames} frames processed.")
        print(f"Saved '{args.output_before}' (original)")
        print(f"Saved '{args.output_after}' (dehazed)")

        # Release resources
        out_before.release()
        out_after.release()
        cap.release()
        cv2.destroyAllWindows()
