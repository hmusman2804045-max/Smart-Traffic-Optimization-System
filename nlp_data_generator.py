import pandas as pd
import random

def generate_nlp_data(output_file="social_media_traffic.csv"):
    print("Generating synthetic social media data...")
    data = []
    normal_tweets = [
        "The commute this morning was surprisingly smooth.",
        "Traffic is moving fine on the downtown expressway.",
        "Just another day on the road, everything looks clear.",
        "No delays reported today. Driving is easy.",
        "Good morning! Roads are empty and the sun is out.",
        "Passed through the city center, no traffic jams at all.",
        "I love it when there's no traffic on my way to work.",
        "Cruising along the highway with zero issues.",
        "The drive was fast today. Rare but nice!",
        "Traffic flow is perfectly normal right now."
    ]
    accident_tweets = [
        "Avoid Main Street, there's a huge crash near the park!",
        "Car accident on the bridge is causing massive delays.",
        "Ambulances just passed by, looks like a bad accident on I-95.",
        "Stay away from 5th Ave, a three-car pileup just happened.",
        "Major crash reported at the intersection. Traffic is stuck.",
        "Emergency services are on site for a collision on the highway.",
        "Traffic is backed up for miles due to an accident.",
        "Huge wreck on the bypass, use an alternative route.",
        "Accident alert! The north lane is completely blocked.",
        "Stuck in traffic because of a crash ahead of me."
    ]
    roadwork_tweets = [
        "Construction work on the tunnel is slowing everyone down.",
        "Expect delays on the outer ring due to road repairs.",
        "New roadwork started on Oak Street. Traffic is heavy.",
        "Narrower lanes today because of the construction crew.",
        "Traffic is crawling because they are repaving the road.",
        "Road closure on 2nd street for bridge maintenance.",
        "Be careful, workers are near the road on the highway exit.",
        "Infrastructure repairs are causing a lot of congestion here.",
        "Cones everywhere! Roadwork is making the drive terrible.",
        "The exit is closed for maintenance until tomorrow."
    ]
    weather_tweets = [
        "Heavy rain is making visibility very poor, drive slow!",
        "Traffic is slow because the roads are flooded.",
        "Snowy conditions are causing slippery roads and delays.",
        "The fog is so thick! Traffic is barely moving.",
        "Stormy weather is causing some chaos on the roads.",
        "Icy roads reported, please be careful out there.",
        "High winds are affecting the traffic flow on the bridge.",
        "Dazzling sun is causing a lot of glare and slow traffic.",
        "Monsoon rains have caused a huge traffic jam.",
        "Be alert! Slick roads after the first rain of the season."
    ]
    for t in normal_tweets: data.append({"text": t, "label": 0})
    for t in accident_tweets: data.append({"text": t, "label": 1})
    for t in roadwork_tweets: data.append({"text": t, "label": 2})
    for t in weather_tweets: data.append({"text": t, "label": 3})
    final_data = data * 5
    random.shuffle(final_data)
    df = pd.DataFrame(final_data)
    df.to_csv(output_file, index=False)
    print(f"Synthetic NLP data saved to: {output_file}")
    print(f"Total tweets generated: {len(df)}")
    return df

if __name__ == "__main__":
    generate_nlp_data()
