import cv2
import numpy as np

# Open the video file
video = cv2.VideoCapture("GX010048.MP4")

# Check if video opened successfully
if not video.isOpened():
    print("Error: Could not open video file")
    exit()

# Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video properties:")
print(f"  FPS: {fps}")
print(f"  Frame count: {frame_count}")
print(f"  Resolution: {width}x{height}")

# Define time range for haze field calculation
haze_start_time = 20  # 20 seconds
haze_end_time = 80    # 1 minute 20 seconds (80 seconds)

# Calculate frame numbers
haze_start_frame = int(haze_start_time * fps)
haze_end_frame = int(haze_end_time * fps)

print(f"\nStep 1: Calculating haze field from frames {haze_start_frame} to {haze_end_frame}")

# Set video to start frame
video.set(cv2.CAP_PROP_POS_FRAMES, haze_start_frame)

# Initialize accumulator for averaging
frame_sum = None
frame_counter = 0

# Process frames in the specified range
current_frame = haze_start_frame
while current_frame <= haze_end_frame:
    ret, frame = video.read()

    if not ret:
        print(f"Warning: Could not read frame {current_frame}")
        break

    # Convert to float for accurate averaging
    frame_float = frame.astype(np.float64)

    # Initialize or accumulate
    if frame_sum is None:
        frame_sum = frame_float
    else:
        frame_sum += frame_float

    frame_counter += 1
    current_frame += 1

    # Print progress every 100 frames
    if frame_counter % 100 == 0:
        print(f"Processed {frame_counter} frames...")

# Calculate average (haze field) - keep as float for precision
if frame_counter > 0:
    haze_field = frame_sum / frame_counter
    print(f"\nHaze field calculated from {frame_counter} frames")

    # Save the haze field visualization
    cv2.imwrite('haze_field.png', haze_field.astype(np.uint8))
    print("Haze field saved as 'haze_field.png'")
else:
    print("Error: No frames were processed")
    video.release()
    exit()

# Step 2: Apply dehazing to video segment (10:10 to 10:30)
dehaze_start_time = 10 * 60 + 10  # 10 minutes 10 seconds = 610 seconds
dehaze_end_time = 10 * 60 + 30    # 10 minutes 30 seconds = 630 seconds

dehaze_start_frame = int(dehaze_start_time * fps)
dehaze_end_frame = int(dehaze_end_time * fps)

print(f"\nStep 2: Processing video segment from {dehaze_start_time}s to {dehaze_end_time}s")
print(f"Frame range: {dehaze_start_frame} to {dehaze_end_frame}")

# Set up video writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_before = cv2.VideoWriter('output_before.mp4', fourcc, fps, (width, height))
out_after = cv2.VideoWriter('output_after.mp4', fourcc, fps, (width, height))

# Dehazing parameter
k = 0.5  # k <= 1

# Set video to dehaze start frame
video.set(cv2.CAP_PROP_POS_FRAMES, dehaze_start_frame)

current_frame = dehaze_start_frame
processed_frames = 0

while current_frame <= dehaze_end_frame:
    ret, frame = video.read()

    if not ret:
        print(f"Warning: Could not read frame {current_frame}")
        break

    # Save original frame
    out_before.write(frame)

    # Apply dehazing: dehazed = frame - k * haze_field
    frame_float = frame.astype(np.float64)
    dehazed = frame_float - k * haze_field

    # Clip values to valid range [0, 255]
    dehazed = np.clip(dehazed, 0, 255).astype(np.uint8)

    # Save dehazed frame
    out_after.write(dehazed)

    processed_frames += 1
    current_frame += 1

    # Print progress
    if processed_frames % 100 == 0:
        print(f"Processed {processed_frames} frames...")

print(f"\nProcessing complete! {processed_frames} frames processed.")
print(f"Saved 'output_before.mp4' (original)")
print(f"Saved 'output_after.mp4' (dehazed with k={k})")

# Release resources
out_before.release()
out_after.release()
video.release()
cv2.destroyAllWindows()