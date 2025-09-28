from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

def fetch_youtube_transcript(video_id):
    """
    Fetches the transcript for a given YouTube video ID.

    Args:
        video_id (str): The YouTube video ID.
        """
    try:
        you_tube_api=YouTubeTranscriptApi()
        youtube_transcript = you_tube_api.fetch(video_id, languages=['en'])
        # convert to raw data and extract text  
        transcript_data=youtube_transcript.to_raw_data()
        transcript=" ".join(chunk['text'] for chunk in transcript_data)
        return transcript                # transcript_data
    
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "No English transcript found for this video."
    except VideoUnavailable:
        return "Video is unavailable."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# print(fetch_youtube_transcript("LPZh9BOjkQs"))
# print(fetch_youtube_transcript("Gfr50f6ZBvo"))