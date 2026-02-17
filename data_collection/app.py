import base64
import csv
import os
import random
import time
import streamlit as st
import config
import consent as raw
import datetime

global_flag = True

# =============================================================================
# Helper Functions
# =============================================================================
def init_session_state():
    """Initialize session state variables on first run."""
    if "phase" not in st.session_state:
        st.session_state.phase = "consent"  # Phases: consent, demographic, begin, baseline, video, test, break, done
    if "consent_given" not in st.session_state:
        st.session_state.consent_given = False
    if "participant_id" not in st.session_state:
        count = len(os.listdir("data")) - 1
        st.session_state.participant_id = count
        print(f"Participant ID: {st.session_state.participant_id}")
    if "demographics" not in st.session_state:
        st.session_state.demographics = {}
    if "current_video" not in st.session_state:
        st.session_state.current_video = None
    if "unseen_videos" not in st.session_state:
        # Define list of videos with durations (in seconds) and their question sets.
        st.session_state.unseen_videos = config.UNSEEN_VIDEOS
    if "cycle_events" not in st.session_state:
        st.session_state.cycle_events = []  # For recording events per cycle

def log_event(event_type, details):
    """Append an event record to the CSV file."""
    filename = f"data/{st.session_state.participant_id}/timestamped_events.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    header = ["participant_id", "event_type", "timestamp", "details"]
    new_row = [st.session_state.participant_id, event_type, time.time(), details]
    file_exists = os.path.exists(filename)
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(new_row)

def get_video_base64(path):
    """Return the base64 encoded string of the video file."""
    with open(path, "rb") as video_file:
        data = video_file.read()
    return base64.b64encode(data).decode("utf-8")

# =============================================================================
# Page Functions
# =============================================================================
#def show_consent_page():
#    st.title("Consent Form")
#    st.write("Please read the following consent form and indicate your agreement to participate in this experiment.")
#    st.write("**Consent:** By checking the box below, you agree to participate in this study and allow us to collect usage data for research purposes.")
#    consent = st.checkbox("I consent to participate")
#    if st.button("Continue"):
#        if consent:
#            st.session_state.consent_given = True
#            log_event("Consent", "User consent given")
#            st.session_state.phase = "demographic"
#            st.rerun()
#        else:
#            st.error("You must consent to continue.")

#Function to display the consent page with signature placeholder and today's date
def show_consent_page():
    st.title("Consent Form")
    st.write("Please read the following consent form and indicate your agreement to participate in this experiment.")
    
    # Display the consent form content
    st.markdown(raw.get_consent_form_content(), unsafe_allow_html=True)

    # Placeholder for participant's signature
    signature = st.text_input("Participant’s Signature (Type your name as a signature)")

    # Automatically select today's date
    today_date = datetime.date.today().strftime("%B %d, %Y")
    st.write(f"**Date:** {today_date}")

    # Consent checkbox
    consent = st.checkbox("I consent to participate")

    # Continue button
    if st.button("Continue"):
        if consent and signature.strip():
            st.session_state.consent_given = True
            log_event("Consent", f"User consent given with signature: {signature}")
            st.session_state.phase = "demographic"
            st.rerun()
        elif not signature.strip():
            st.error("You must provide your signature to continue.")
        else:
            st.error("You must consent to continue.")


def show_demographic_page():
    
    st.write("### Demographic Information")
    st.write("Please provide some basic demographic information.")
    
    # Basic Demographic Information
    age = st.number_input("Age:", min_value=0, max_value=100, step=1)
    gender = st.selectbox("Gender:", ["Prefer not to say", "Female", "Male", "Other"])
    education = st.text_input("Highest level of education:")
    schooling = st.text_input("What was the medium of instruction at your school?")
    city = st.text_input("In which city have you been living for the past two years?")
    subjects = st.text_input("What subjects did you take in your 11th and 12th grade?")

    # Perception Questions
    st.write("### Perception of Knowledge in Different Areas")

    perception_data = {}

    # 1. Computational & Data Science
    perception_data["computational_science"] = st.radio(
        "How would you rate your understanding of computational and data science concepts, "
        "including machine learning, deep learning, and computer vision?",
        options=["1 - No prior knowledge", "2 - Heard of them, but don’t fully understand",
                 "3 - Understand some concepts but struggle with applications",
                 "4 - Comfortable with these topics, can apply them",
                 "5 - Strong grasp, can analyze complex problems"],
        index=0
    )

    # 2. Biological Sciences
    perception_data["biological_sciences"] = st.radio(
        "How confident are you in your knowledge of biological sciences, particularly genetics, DNA, genes, and protein synthesis?",
        options=["1 - No background", "2 - Basic concepts, lack deeper understanding",
                 "3 - General understanding, need more clarity",
                 "4 - Comfortable with most concepts",
                 "5 - Strong understanding, can explain to others"],
        index=0
    )

    # 3. Theoretical & Applied Mathematics
    perception_data["applied_mathematics"] = st.radio(
        "How would you rate your familiarity with theoretical and applied mathematics, including fluid mechanics and quantum mechanics?",
        options=["1 - Completely new", "2 - Some exposure, struggle with equations",
                 "3 - Understand principles, need more practice",
                 "4 - Comfortable, can solve related problems",
                 "5 - Deep understanding, can apply effectively"],
        index=0
    )

    # Initialize session state flags
    if "pageflag" not in st.session_state:
        st.session_state.pageflag = False
        st.session_state.videoflag = False
        st.session_state.testflag = False
        st.session_state.breakflag = False

    # Submit Button
    if st.button("Submit"):
    
        st.session_state.demographics = {
            "age": age, "gender": gender, "education": education,
            "schooling": schooling, "city": city, "subjects": subjects,
            "perception": perception_data
        }

        # Logging the event with perception data
        log_event("Demographics",
            f"Age: {age}, Gender: {gender}, Education: {education}, "
            f"Schooling: {schooling}, City: {city}, Subjects: {subjects}, "
            f"Perception - Computational Science: {perception_data['computational_science']}, "
            f"Biological Sciences: {perception_data['biological_sciences']}, "
            f"Applied Mathematics: {perception_data['applied_mathematics']}"
        )

        st.session_state.phase = "begin"
        st.rerun()

def show_begin_page():
    st.title("Experiment Start")
    st.write("Click the 'Begin' button to start.")
    bt = st.empty()
    if bt.button("Begin"):
        st.session_state.begin_time = time.time()
        log_event("Begin", f"Begin clicked at {st.session_state.begin_time}")
        bt.empty()
        # Show instructions after clicking "Begin"
        placeholder3 = st.empty()
        placeholder4 = st.empty()
        placeholder5 = st.empty()
        placeholder6 = st.empty()
      
         # Show instructions after clicking "Begin"
         # Show instructions using placeholders
        placeholder3.write("### Important Instructions while waiting:")
        placeholder4.write("- Please **avoid excessive head movements** during the experiment for accurate measurements.")
        placeholder5.write("- Try to **stay as still as possible** and **minimize distractions**.")
        placeholder6.write("- Keep your **focus on the screen** rather than looking around.")


        # Countdown Timer for 30 seconds
        for remaining in range(30, 0, -1):
            text = st.empty()
            text.write(f"Transitioning to baseline in **{remaining} seconds**...")
            time.sleep(1)
            text.empty()
            
            
#            st.rerun()  # Refresh countdown dynamically
        placeholder3.empty()
        placeholder4.empty()
        placeholder5.empty()
        placeholder6.empty()
        st.empty()
        # After 30 seconds, transition to baseline
        st.session_state.phase = "baseline"
        st.rerun()


def show_baseline_page():
    global global_flag
    st.title("Baseline Questions")
    # Ensure current_video is selected
    if st.session_state.current_video is None:
        st.session_state.current_video = random.choice(st.session_state.unseen_videos)
    print("Checkpoint: Reached baseline page video :", st.session_state.current_video["filename"])
    
    if "baseline_start_time" not in st.session_state or st.session_state.pageflag:
        st.session_state.baseline_start_time = time.time()
        log_event("Baseline Start", f"Baseline QA started at {st.session_state.baseline_start_time} for video {st.session_state.current_video['filename']}")
        st.session_state.pageflag = False

    video_info = st.session_state.current_video
    responses = {}
    
    #Doing this so that placeholders is not instantiated multiple times
    if global_flag:
        placeholders = [st.empty() for _ in video_info["baseline_questions"]]
        global_flag = False
    
    # Assign each question to a different placeholder
    for i, (placeholder1, q) in enumerate(zip(placeholders, video_info["baseline_questions"]), start=1):
        with placeholder1:
            responses[f"baseline_{i}"] = st.radio(
                q["question"], q["options"],
                key=f"baseline_{video_info['filename']}_{i}"
        )
            
    elapsed = time.time() - st.session_state.baseline_start_time
    remaining = int(75 - elapsed)
    placeholder = st.empty()
    placeholder.write(f"**Timer: {remaining} seconds remaining for baseline questions.**")

    # Automatically proceed when time runs out (no "Submit" button).
    if remaining <= 0:
        st.session_state.baseline_end_time = time.time()
        log_event("Baseline End", f"Baseline QA ended at {st.session_state.baseline_end_time} for video {video_info['filename']}")
        log_event("Baseline Responses", f"Responses: {responses}")
        placeholder.empty()
        for placeholder1 in placeholders:
            placeholder1.empty()
        st.session_state.baseline_responses = responses
        st.session_state.phase = "video"
        global_flag = True
        st.rerun()
    else:
        time.sleep(1)
        st.rerun()

def show_video_page():
    # Ensure current_video is selected
    if st.session_state.current_video is None:
        st.session_state.current_video = random.choice(st.session_state.unseen_videos)
    # Ensure video_start_time is initialized
    if "video_start_time" not in st.session_state or st.session_state.videoflag:
        st.session_state.video_start_time = time.time()
        st.session_state.videoflag = False
        log_event("Video Start", f"Video {st.session_state.current_video['filename']} started at {st.session_state.video_start_time}")
    video_info = st.session_state.current_video
    st.title("Video Playback")
    try:
        video_b64 = get_video_base64(video_info["filename"])
        video_html = f"""
            <video width="640" height="480" autoplay controls style="display:block; margin:auto;">
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        """
        st.components.v1.html(video_html, height=500)
    except Exception as e:
        st.error(f"Error loading video: {e}")

    elapsed = time.time() - st.session_state.video_start_time
    remaining = int(video_info["duration"] - elapsed)
    placeholder = st.empty()
    placeholder.write(f"**Timer: {remaining} seconds remaining for video.**")

    # Automatically proceed once the video duration is over.
    if elapsed >= video_info["duration"]:
        st.session_state.video_end_time = time.time()
        log_event("Video End", f"Video {video_info['filename']} ended at {st.session_state.video_end_time}")
        placeholder.empty()
        st.session_state.phase = "test"
        st.rerun()
    else:
        time.sleep(1)
        st.rerun()

def show_test_page():
    global global_flag
    st.title("Test Questions")
    
    #st.session_state.testflag makes sure that we don't skip baseline, video and test questions cycle when you come back again for the second video
    if "test_start_time" not in st.session_state or st.session_state.testflag:
        st.session_state.test_start_time = time.time()
        st.session_state.testflag = False
        log_event("Test Start", f"Test QA started at {st.session_state.test_start_time} for video {st.session_state.current_video['filename']}")
    video_info = st.session_state.current_video
    responses = {}
    placeholder2 = st.empty()
    
    if global_flag:
        placeholders = [st.empty() for _ in video_info["after_questions"]]
        global_flag = False
    
        # Assign each question to a different placeholder
    for i, (placeholder1, q) in enumerate(zip(placeholders, video_info["after_questions"]), start=1):
        with placeholder1:
            responses[f"test_{i}"] = st.radio(
                q["question"], q["options"],
                key=f"test_{video_info['filename']}_{i}"
        )
            
    elapsed = time.time() - st.session_state.test_start_time
    remaining = int(95 - elapsed)
    placeholder = st.empty()
    placeholder.write(f"**Timer: {remaining} seconds remaining for test questions.**")

    # Automatically proceed when time runs out (no "Submit" button).
    if remaining <= 0:
        st.session_state.test_end_time = time.time()
        log_event("Test End", f"Test QA ended at {st.session_state.test_end_time} for video {video_info['filename']}")
        log_event("Test Responses", f"Responses: {responses}")
        st.session_state.test_responses = responses
        placeholder.empty()
        for placeholder1 in placeholders:
            placeholder1.empty()
        global_flag = True
        # ALWAYS go to break, do NOT check if only 1 video left here
        st.session_state.phase = "break"
        st.rerun()
    else:
        time.sleep(1)
        st.rerun()

def show_break_page():
    st.title("Break")
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder1.write("Please wait, taking a short 30-second break...")
    
    placeholder3 = st.empty()
    placeholder4 = st.empty()
    placeholder5 = st.empty()
    placeholder6 = st.empty()
  
     # Show instructions after clicking "Begin"
     # Show instructions using placeholders
    placeholder3.write("### Important Instructions while waiting:")
    placeholder4.write("- Please **avoid excessive head movements** during the experiment for accurate measurements.")
    placeholder5.write("- Try to **stay as still as possible** and **minimize distractions**.")
    placeholder6.write("- Keep your **focus on the screen** rather than looking around.")

    if "break_start_time" not in st.session_state or st.session_state.breakflag:
        st.session_state.break_start_time = time.time()
        st.session_state.breakflag = False
        log_event("Break Start", f"Break started at {st.session_state.break_start_time}")
    elapsed = time.time() - st.session_state.break_start_time
    placeholder2.write(f"**Timer: {int(30 - elapsed)} seconds remaining for break.**")

    if elapsed < 30:
        time.sleep(1)
        st.rerun()
    else:
        st.session_state.break_end_time = time.time()
        log_event("Break End", f"Break ended at {st.session_state.break_end_time}")
        # Remove the current video from unseen_videos
        st.session_state.unseen_videos = [
            vid for vid in st.session_state.unseen_videos
            if vid["filename"] != st.session_state.current_video["filename"]
        ]
        placeholder1.empty()
        placeholder2.empty()
        placeholder3.empty()
        placeholder4.empty()
        placeholder5.empty()
        placeholder6.empty()
        # Reset cycle-specific variables so leftover timers won't appear
        # for key in [
        #     "begin_time", "baseline_start_time", "baseline_end_time",
        #     "video_start_time", "video_end_time", "test_start_time",
        #     "test_end_time", "break_start_time", "break_end_time"
        # ]:
        #     if key in st.session_state:
        #         del st.session_state[key]
        st.session_state.current_video = None

        # If there are still unseen videos, go to baseline; otherwise, finish
        if st.session_state.unseen_videos:
            st.session_state.phase = "baseline"
            st.session_state.pageflag = True
            st.session_state.videoflag=True
            st.session_state.testflag=True
            st.session_state.breakflag=True
        else:
            st.session_state.phase = "done"
        st.rerun()

def show_done_page():
    st.title("Experiment Complete")
    st.write("Thank you for participating!")
    st.write("Your responses and timings have been recorded.")

# =============================================================================
# Main App Flow
# =============================================================================
init_session_state()

if st.session_state.phase == "consent":
    show_consent_page()
elif st.session_state.phase == "demographic":
    show_demographic_page()
elif st.session_state.phase == "begin":
    show_begin_page()
elif st.session_state.phase == "baseline":
    print("Checkpoint: Reached baseline page")
    # if st.session_state.current_video is None:
    #     st.session_state.current_video = random.choice(st.session_state.unseen_videos)
    show_baseline_page()
elif st.session_state.phase == "video":
    show_video_page()
elif st.session_state.phase == "test":
    show_test_page()
elif st.session_state.phase == "break":
    show_break_page()
elif st.session_state.phase == "done":
    show_done_page()
