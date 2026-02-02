import matplotlib.pyplot as plt  # Import for data visualization
import pandas as pd
from ultralytics import YOLO

# --- CONFIGURATION ---
# Load the pre-trained YOLOv8 model for inference.
# 'yolov8n.pt' is used for rapid, lightweight processing.
model = YOLO("yolov8n.pt")
# Define the input source path for video stream analysis.
source_video_path = "data/videos/video.mp4"

# Dictionary used to aggregate the total number of frames each unique object ID has been visible.
# This metric forms the basis for Dwell Time calculation.
# Key: Unique Object Tracking ID (int), Value: Accumulated Frame Count (int)
tracked_objects_counts = {}

# System-defined Frames Per Second (FPS) of the source video stream.
# This value is critical for converting frame counts to accurate time (seconds).
FPS = 30

# --- EXECUTION: REAL-TIME TRACKING AND DATA COLLECTION ---
print("Starting real-time tracking and data collection...")

# Initiate the tracking pipeline using Ultralytics' integrated function.
# The 'tracker' argument specifies the use of the ByteTrack algorithm for persistent object ID assignment.
results = model.track(
    source=source_video_path,
    tracker="bytetrack.yaml",
    stream=True,  # Enables processing frames sequentially for video streaming
    conf=0.3,  # Sets the minimum confidence threshold for detection inclusion
    save=True,
)  # Flag to output the processed video with tracking overlays

for frame_idx, result in enumerate(results):
    # Conditional check to ensure tracking data (boxes.id) is available for the current frame.
    if result.boxes is not None and result.boxes.id is not None:
        # Extract the list of unique, persistent Track IDs identified in the current frame.
        track_ids = result.boxes.id.int().cpu().tolist()

        # Iterate through the observed IDs and update the global frame counter.
        for track_id in track_ids:
            tracked_objects_counts[track_id] = tracked_objects_counts.get(track_id, 0) + 1

# --- STATISTICAL ANALYSIS AND REPORTING ---
print("\n--- Statistical Analysis (Dwell Time Report) ---")

# 1. Prepare Data for Export
report_data = []

# Process the aggregated frame counts to calculate final Dwell Time.
for obj_id, frame_count in tracked_objects_counts.items():
    # Calculate the total observation time in seconds using the defined FPS.
    dwell_time_seconds = frame_count / FPS

    # Output the calculated metric to the standard console for real-time validation.
    print(f"Object ID #{obj_id} was observed for {frame_count} frames, equating to {dwell_time_seconds:.2f} seconds.")

    # Structure the collected data into a list of dictionaries for Pandas DataFrame creation.
    report_data.append(
        {"Object_ID": obj_id, "Frames_Observed": frame_count, "Dwell_Time_Seconds": round(dwell_time_seconds, 2)}
    )

# 2. Export to CSV using Pandas
df = pd.DataFrame(report_data)
output_csv_path = "tracking_report.csv"
# Write the DataFrame to a CSV file. 'index=False' prevents the row indices from being written.
df.to_csv(output_csv_path, index=False)

print("\n--- Analysis Complete ---")
print(f"REPORT SAVED SUCCESSFULLY to: {output_csv_path}")


# --- DATA VISUALIZATION (Top 5 Dwell Time) ---
print("\n--- Generating Visualization ---")

# Sort the DataFrame by Dwell Time in descending order and select the top 5 records for charting.
df_top5 = df.sort_values(by="Dwell_Time_Seconds", ascending=False).head(5)

# Initialize the Matplotlib figure with specified dimensions.
plt.figure(figsize=(10, 6))
# Create a bar chart comparing Object ID against its total Dwell Time.
plt.bar(df_top5["Object_ID"].astype(str), df_top5["Dwell_Time_Seconds"], color="skyblue")

# Set chart metadata for clarity and professionalism.
plt.title("Top 5 Observed Objects (Dwell Time)")
plt.xlabel("Object ID")
plt.ylabel("Dwell Time (Seconds)")
# Add a horizontal grid for easier data comparison.
plt.grid(axis="y", linestyle="--")

# Save the generated chart to the project directory in PNG format.
output_image_path = "dwell_time_top5_chart.png"
plt.savefig(output_image_path)
# Close the figure explicitly to manage memory consumption.
plt.close()

print(f"CHART SAVED SUCCESSFULLY to: {output_image_path}")
