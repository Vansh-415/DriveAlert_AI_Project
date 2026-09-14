import time, av, streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from tensorflow.keras.models import load_model
from alert import AlertManager
from detector import DrowsinessDetector

st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗", layout="wide")
st.title("🚗 Driver Drowsiness Detection System")

@st.cache_resource
def get_detector():
    return DrowsinessDetector(load_model("drowsiness_model.h5"))

detector = get_detector()
if "alert_mgr" not in st.session_state:
    st.session_state.alert_mgr = AlertManager()
alert_mgr = st.session_state.alert_mgr

st.sidebar.title("System Monitor")
status_box = st.sidebar.empty()
episode_box = st.sidebar.empty()
st.sidebar.markdown("---")
st.sidebar.caption(
    "**How it works:**\n"
    "1. Detects face and isolates left & right eye regions in the upper face band.\n"
    "2. Classifies eye states using CNN.\n"
    "3. Rolling 20-frame buffer filters normal blinks.\n"
    "4. If ≥15/20 frames indicate drowsiness, sound plays continuously until awake."
)
audio_box = st.empty()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(detector.process_frame(img), format="bgr24")

ctx = webrtc_streamer(
    key="drowsiness-streamer",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
)

# Active polling loop: updates monitor and manages continuous alarm audio
while ctx.state.playing:
    status = detector.smoothed_state
    color = "#ff4757" if status == "Drowsy" else "#ffa502" if status == "Warning" else "#2ed573"
    status_box.markdown(f"### Status: <span style='color:{color}'>{status.upper()}</span>", unsafe_allow_html=True)
    episode_box.metric("Drowsy Episodes", alert_mgr.total_episodes)

    # Audio control: play continuously while Drowsy; stop immediately once awake
    action = alert_mgr.update(status == "Drowsy")
    if action == "start":
        audio_box.empty()
        audio_box.audio(alert_mgr.audio_bytes, format="audio/wav", autoplay=True, loop=True)
    elif action == "stop":
        audio_box.empty()

    time.sleep(0.1)

if not ctx.state.playing:
    status_box.markdown("### Status: <span style='color:#747d8c'>OFFLINE</span>", unsafe_allow_html=True)
    episode_box.metric("Drowsy Episodes", alert_mgr.total_episodes)
