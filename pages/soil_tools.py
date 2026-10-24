"""
Soil tools page: response-surface sampling design, soil-property calibration
against EC and fuzzy c-means management zones, on a survey loaded on the
survey-analysis page (with its corrections) or the sample survey.
"""
from pathlib import Path

import streamlit as st

import pipeline
import RIs_v2 as R
import ui_tools

st.title("Soil tools")
st.caption(
    "Sampling design, soil-property calibration and management zones from the EC channels of "
    "a GEM-2 survey. " + ui_tools.doc_link("soil-tools", "", "Soil tools")
)

surveys = dict(st.session_state.get(R.SURVEYS_KEY) or {})
if not surveys and st.session_state.get(R.SAMPLE_STATE_KEY):
    sample = R.SampleFile.load()
    surveys = {sample.name: (sample.getvalue(), sample.name, pipeline.PrepSettings())}

if not surveys:
    st.info("Upload a GEM-2 survey on the **Survey analysis** page first: the soil tools use it "
            "with the corrections and settings chosen there.")
    if R.SAMPLE_FILE.exists() and st.button("Try the sample survey", key="soil_sample",
                                             icon=":material/science:"):
        st.session_state[R.SAMPLE_STATE_KEY] = True
        st.rerun()
    st.stop()

name = st.selectbox("Survey", list(surveys), key="soil_survey") if len(surveys) > 1 else next(iter(surveys))
data, file_name, prep = surveys[name]
try:
    table, _ = R.prepared_table(data, file_name, prep)
except ValueError as exc:
    st.error(f"Could not prepare {file_name}: {exc}")
    st.stop()
st.subheader(file_name)
ui_tools.render_soil_tools(table, Path(file_name).stem)
