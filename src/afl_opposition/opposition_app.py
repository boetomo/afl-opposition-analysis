"""Streamlit interface for archived event forecasts and opposition briefings."""
from datetime import timedelta
from pathlib import Path
import os
import re

import pandas as pd
import streamlit as st

from afl_opposition.opposition import (
    load_bundle, clubs, select_matches, review_rows, tendencies, predictions,
    pitch, label, clock, report_html, PROBS,
)


@st.cache_resource(show_spinner='Loading match archive…')
def cached_bundle(folder, stamp):
    return load_bundle(folder)


@st.cache_data(show_spinner=False)
def briefing(folder, stamp, club, cutoff, ids):
    bundle = cached_bundle(folder, stamp)
    matches = select_matches(bundle, club, cutoff, 30)
    matches = matches.loc[matches.match_id.isin(ids)]
    return report_html(bundle, matches, club, cutoff)


def main(root):
    root = Path(root)
    st.set_page_config(page_title='AFL Opposition Analysis', page_icon='🏉', layout='wide')
    st.markdown('''<style>
    .stApp {background:#f7f9f9} .block-container {padding-top:4.8rem;max-width:1500px}
    h1,h2,h3 {color:#173a45} [data-testid="stSidebar"] {background:#edf3f2}
    [data-testid="stMetric"] {background:white;border:1px solid #e1e9e7;border-radius:10px;padding:12px 18px}
    .eyebrow {font-size:12px;font-weight:700;letter-spacing:2px;color:#0c8378;text-transform:uppercase}
    </style>''', unsafe_allow_html=True)
    options = {}
    configured = os.environ.get('AFL_APP_BUNDLE')
    if configured:
        options['Configured archive'] = Path(configured)
    else:
        for title, relative in [('2026 archive', 'artifacts/opposition-app-v1'), ('Sample archive · all clubs', 'reports/examples/opposition-demo')]:
            folder = root / relative
            if (folder / 'manifest.json').is_file():
                options[title] = folder
    st.sidebar.title('Match selection')
    if not options:
        st.title('AFL Opposition Analysis')
        st.info('No match bundle is available. Prepare the local archive or include the example bundle.')
        st.code('python -m afl_opposition.prepare_opposition', language='shell')
        st.stop()
    source = st.sidebar.selectbox('Data', list(options), key='source')
    folder = options[source]
    try:
        stamp = (folder / 'manifest.json').stat().st_mtime_ns
        bundle = cached_bundle(str(folder), stamp)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Could not load the match bundle: {type(exc).__name__}. Check the setup instructions in docs/app.md.')
        st.stop()
    names = clubs(bundle)
    club = st.sidebar.selectbox('Team to analyse', list(names), format_func=names.get, key='club',
        help='Filters the team’s matches and sets the pitch orientation. Viewing the same match event from either team gives the same forecast.')
    end = max(bundle.matches.date) + timedelta(days=1)
    cutoff = st.sidebar.date_input('Matches before', value=end, key='cutoff', help='Only matches strictly before this date are included. This is historical archive review.')
    count = st.sidebar.number_input('Recent matches', min_value=1, max_value=30, value=5, step=1, key='count')
    matches = select_matches(bundle, club, cutoff, int(count))
    st.sidebar.caption('The saved model was trained on 2023–2025. Available replay matches are from 2026. No model trains while this app is open.')
    st.markdown('<div class="eyebrow">AFL Opposition Analysis</div>', unsafe_allow_html=True)
    st.title(f'{names[club]} · opposition review')
    st.write('Explore the play, inspect what the model expects next, and build a briefing from the evidence.')
    if bundle.manifest['coverage'] == 'sample':
        st.info(bundle.manifest['coverage_note'])
    else:
        st.caption('Historical archive · frozen decoder · predictions based only on the observed quarter history')
    if matches.empty:
        st.info('No recorded matches for this team before that date. Choose a later date or a different archive.')
        st.stop()
    rows = review_rows(bundle, matches, club)
    metrics = st.columns(3)
    metrics[0].metric('Matches selected', len(matches))
    metrics[1].metric('Recorded quarters', int(matches.quarters_available.sum()))
    metrics[2].metric('Eligible team transitions', f'{len(rows):,}')
    replay_tab, report_tab = st.tabs(['Event replay & predictions', 'Opposition briefing'])
    with replay_tab:
        col1, col2 = st.columns([4, 1])
        by_id = matches.set_index('match_id')
        match_id = col1.selectbox('Match', matches.match_id.tolist(), key='match',
            format_func=lambda mid:f'{by_id.loc[mid,"date"]} · {by_id.loc[mid,"home_name"]} v {by_id.loc[mid,"away_name"]} · R{by_id.loc[mid,"round_number"]}')
        match = by_id.loc[match_id]
        match_events = bundle.events.loc[bundle.events.match_id.eq(match_id)]
        period = col2.selectbox('Quarter', sorted(match_events.period.unique().tolist()), key='quarter')
        quarter = match_events.loc[match_events.period.eq(period)].reset_index(drop=True)
        context = (source, club, str(cutoff), match_id, int(period))
        if st.session_state.get('replay_context') != context:
            st.session_state.update(replay_context=context, event_number=1, playing=False, reveal=True)
        if st.session_state.get('playing') and st.session_state.get('event_number', 1) >= len(quarter):
            st.session_state.playing = False

        def pause():
            st.session_state.playing = False

        def move(delta):
            st.session_state.event_number = min(len(quarter), max(1, st.session_state.event_number + delta))
            pause()

        def toggle():
            if st.session_state.event_number >= len(quarter):
                st.session_state.event_number = 1
            st.session_state.playing = not st.session_state.get('playing', False)

        controls = st.columns([1, 1, 1, 4])
        controls[0].button('Previous', on_click=move, args=(-1,), key='previous', width='stretch')
        controls[1].button('Pause' if st.session_state.get('playing') else 'Play', on_click=toggle, key='play', width='stretch', type='primary')
        controls[2].button('Next', on_click=move, args=(1,), key='next', width='stretch')
        controls[3].checkbox('Reveal the actual next event', key='reveal')

        @st.fragment(run_every=.85 if st.session_state.get('playing') else None)
        def replay():
            if st.session_state.get('playing'):
                st.session_state.event_number = min(len(quarter), st.session_state.event_number + 1)
                if st.session_state.event_number == len(quarter):
                    st.session_state.playing = False
                    st.rerun()
            st.slider('Observed events in this quarter', min_value=1, max_value=len(quarter), key='event_number', on_change=pause)
            position = st.session_state.event_number - 1
            row = quarter.iloc[position]
            current_code = int(row.side) * 12 + list(bundle.manifest['actions']).index(row.action_type) if row.side < 2 else 24
            current_label = 'Unknown passage' if row.action_type == 'gap' else label(current_code, match.home_name, match.away_name)
            st.markdown(f'**Q{int(period)} · {clock(row.period_seconds)} · {current_label}**')
            left, right = st.columns([1.65, 1])
            with left:
                st.plotly_chart(pitch(quarter, position, match, club, reveal=st.session_state.reveal), width='stretch',
                    config={'displayModeBar':False}, key='pitch')
                st.caption('Last 12 observations shown; the decoder uses the whole observed quarter. The selected team attacks right. Connecting lines join recorded locations, not a tracked ball flight.')
            with right:
                st.subheader('Next-event forecast for this match')
                st.caption('The model uses both teams and the observed quarter history to predict the next recorded event. For summaries across matches, open Opposition briefing.')
                usable = bool(row.eligible) and row.action_type != 'gap'
                if usable:
                    if position == 0:
                        st.info('This replay starts at the first recorded event of the quarter. With very little play observed, forecasts can look similar across teams. Step forward or press Play to see how they respond to the developing sequence.')
                    predicted = predictions(row, match)
                    shown = predicted.head(6).copy()
                    shown['Probability'] = shown.Probability * 100
                    st.dataframe(shown, hide_index=True, width='stretch',
                        column_config={'Probability':st.column_config.ProgressColumn('Model estimate', min_value=0, max_value=100, format='%.1f%%')})
                    with st.expander('All 25 outcomes'):
                        all_probs = predicted.copy()
                        all_probs['Probability'] = all_probs.Probability.map(lambda x:f'{x:.2%}')
                        st.dataframe(all_probs, hide_index=True, width='stretch')
                    st.caption('The purple cross is one location estimate for an ordinary action. It is shared across possible actions; it is not a predicted destination for each action. Probabilities are model estimates, not guaranteed frequencies.')
                else:
                    st.warning('This observation is flagged for archive review. Its forecast is withheld, and it is excluded from the briefing summaries.')
                if st.session_state.reveal:
                    if position + 1 < len(quarter):
                        nxt = quarter.iloc[position + 1]
                        target_code = int(nxt.side) * 12 + list(bundle.manifest['actions']).index(nxt.action_type) if nxt.side < 2 else 24
                        actual = 'Unknown passage' if nxt.action_type == 'gap' else label(target_code, match.home_name, match.away_name)
                        st.info(f'Actual next · {clock(nxt.period_seconds)} · {actual}')
                        if not row.labelled:
                            st.caption('This transition is excluded from benchmark scoring because of archive quality flags.')
                    else:
                        st.info('End of recorded quarter: there is no next observation to reveal.')
            with st.expander('Observed event list'):
                seen = quarter.iloc[:position + 1][['period_seconds','action_type','side','eligible']].tail(30).copy()
                seen['Time'] = seen.period_seconds.map(clock)
                seen['Team'] = seen.side.map({0:match.home_name,1:match.away_name,2:'Neutral / unknown'})
                seen = seen.rename(columns={'action_type':'Action','eligible':'Archive quality passed'})
                st.dataframe(seen[['Time','Team','Action','Archive quality passed']], hide_index=True, width='stretch')
        replay()
    with report_tab:
        st.subheader('What does this team tend to do next?')
        st.write('Compare recorded outcomes with the model’s average estimates in the same situations. Each row describes the current action and field third.')
        table = tendencies(rows)
        st.dataframe(table.round(1), hide_index=True, width='stretch')
        st.caption('Kick / handball refer to this team’s next recorded action. Other-team refers to the next recorded action belonging to the opposition, not total turnover risk. Other actions and boundaries are not shown in these columns. Small groups are unstable.')
        with st.expander('Match coverage and limitations'):
            st.dataframe(matches[['date','home_name','away_name','quarters_available']], hide_index=True)
            st.write('Retrospective archive quality filters exclude uncertain transitions. The 2026 benchmark is already observed; this is not evidence of live-match reliability. Rare actions remain difficult. This first version does not compare clubs after controlling for their situations or recommend defensive tactics.')
        st.subheader('Download the briefing')
        st.write('A standalone HTML report with the selected matches, tendency tables and up to six example passages. Open it in a browser, or print it to PDF.')
        report_key = (str(folder), stamp, club, str(cutoff), tuple(matches.match_id))
        if st.button('Generate opposition report', type='primary', key='generate'):
            with st.spinner('Building the briefing…'):
                st.session_state.report_html = briefing(str(folder), stamp, club, cutoff, tuple(matches.match_id))
                st.session_state.report_key = report_key
        if st.session_state.get('report_key') == report_key:
            filename = re.sub(r'[^a-z0-9]+','-', names[club].lower()).strip('-') + f'-opposition-{cutoff}.html'
            st.download_button('Download opposition report', st.session_state.report_html, file_name=filename,
                mime='text/html', key='download', on_click='ignore', type='primary')
    st.caption('Model: Tuned decoder-only Transformer · predeclared seed 2026 · trained on 2023–2025 · source: torpdata. This app performs no training.')
