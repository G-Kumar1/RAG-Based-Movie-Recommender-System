def generate_recommendation(movies, preference):
    recommendations = []
    for _, row in movies.iterrows():
        recommendations.append(
            f"🎬 **{row['Series_Title']}**\n"
            f"Why: Matches your preference for {preference}\n"
        )
    return recommendations
