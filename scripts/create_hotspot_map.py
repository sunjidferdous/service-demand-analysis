import pandas as pd
import folium

input_file = "data/processed/dbscan_cluster_summary.csv"
output_file = "data/processed/spatial_hotspot_map.html"

df = pd.read_csv(input_file)

# Map center
center_lat = df["centroid_latitude"].mean()
center_lon = df["centroid_longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=4,
    tiles="OpenStreetMap"
)

# Top 100 clusters by business density
top_clusters = df.nlargest(100, "business_count")

for _, row in top_clusters.iterrows():

    popup_text = f"""
    <b>Cluster ID:</b> {int(row['cluster_id'])}<br>
    <b>Businesses:</b> {int(row['business_count']):,}<br>
    <b>Total Reviews:</b> {int(row['total_review_count']):,}<br>
    <b>Avg Reviews/Business:</b> {row['avg_review_count']:.2f}<br>
    <b>Avg Stars:</b> {row['avg_stars']:.2f}
    """

    folium.CircleMarker(
        location=[
            row["centroid_latitude"],
            row["centroid_longitude"]
        ],
        radius=max(5, min(20, row["business_count"] / 700)),
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=f"Cluster {int(row['cluster_id'])}",
        fill=True
    ).add_to(m)

m.save(output_file)

print("Spatial hotspot map created successfully!")
print(f"Saved to: {output_file}")