"""
Streamlit frontend of the Tangles Explorer prototype.

`main` lays out a three-column page: the left column holds the search setup
(CSV upload, agreement, order function, advanced t-SNE settings, excluded
columns, hover columns, parameter presets), the middle column the tabbed
visualizations, and the right column the interactive rule-list.

Every widget lives in `st.session_state`, so a preset can be saved and restored
by writing the same keys back. On each interaction Streamlit reruns the whole
script, which re-runs the pipeline in `run` and rebuilds all figures — the
`tangle_dict` returned by `core_tangles.build_tangle_model` is the single source
of truth passed to every `vis_*` function.

The `vis_*` functions are the sections of the page: each one renders its own
widgets, reads the current selection from the session state and draws the
matching figure from `viz_plotly` / `viz_tangles_matplotlib`.
"""

import streamlit as st
import pandas as pd
import numpy as np

from core_tangles import build_tangle_model, build_customdata, membership_from_scores, scan_agreement_range_from_tangle_dict
from viz_plotly import (
    fig_tsne_membership,
    fig_tangle_feature_heatmap,
    fig_parallel_coordinates,
    fig_tsne_compare_tangles,
    fig_agreement_scan_comparison,
    fig_tsne_rule_combination,
)

from viz_tangles_matplotlib import fig_tree_of_tangles

# ------------------------
# Session state initialization
# ------------------------

def _init_session_state(default_agreement):
    """
    Seed every session-state key with its default on the first run.

    Each key is only written if absent, so a rerun never overwrites a user's
    widget value. `default_agreement` is derived from the uploaded dataset size
    and therefore only known after the upload.
    """
    if "selected_tangle" not in st.session_state:
        st.session_state["selected_tangle"] = None
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = None
    if "agreement_scan_runs" not in st.session_state:
        st.session_state["agreement_scan_runs"] = []
    if "saved_global_parameter_sets" not in st.session_state:
        st.session_state["saved_global_parameter_sets"] = []
    if "load_global_parameter_set_name" not in st.session_state:
        st.session_state["load_global_parameter_set_name"] = None
    if "selected_rule_ids" not in st.session_state:
        st.session_state["selected_rule_ids"] = []
    if "agreement" not in st.session_state:
        st.session_state["agreement"] = int(default_agreement)
    if "max_number_of_seps" not in st.session_state:
        st.session_state["max_number_of_seps"] = 10
    if "order_function" not in st.session_state:
        st.session_state["order_function"] = "01"
    if "use_order_function" not in st.session_state:
        st.session_state["use_order_function"] = True
    if "perplexity" not in st.session_state:
        st.session_state["perplexity"] = 10
    if "normalized_values_for_tsne" not in st.session_state:
        st.session_state["normalized_values_for_tsne"] = True
    if "max_same_parameter" not in st.session_state:
        st.session_state["max_same_parameter"] = 2
    if "show_only_non_numeric_exclude" not in st.session_state:
        st.session_state["show_only_non_numeric_exclude"] = False
    if "exclude_cols" not in st.session_state:
        st.session_state["exclude_cols"] = []

def _next_run_label(existing_labels: list[str]) -> str:
    """
    Generate the next free default label ("Run 1", "Run 2", ...) for a saved scan.

    Scans the existing labels for the "Run <n>" pattern and returns one past the
    highest number found, ignoring labels the user has renamed. Trailing text
    after the number (e.g. "Run 3 | order=02") is tolerated.
    """
    max_idx = 0

    for label in existing_labels:
        if not label.startswith("Run "):
            continue

        rest = label[4:]
        # read the leading digits after the "Run " prefix
        num_str = ""
        for ch in rest:
            if ch.isdigit():
                num_str += ch
            else:
                break

        if not num_str:
            continue

        try:
            n = int(num_str)
        except ValueError:
            continue

        if n > max_idx:
            max_idx = n

    return f"Run {max_idx + 1}"

def run(uploaded, agreement, max_same_parameter, perplexity,
        max_number_of_seps, order_function, normalized_values_for_tsne, exclude_cols):
    """
    Run the full tangle pipeline for the current widget values.

    Thin adapter around `core_tangles.build_tangle_model`: it forwards the UI
    parameters and translates the widget semantics of `max_same_parameter`
    (number of partitions per numeric feature, minimum 2) into the number of
    thresholds the pipeline expects (`k_per_feature`, one less than that).

    Returns
    -------
    dict
        The `tangle_dict` consumed by all `vis_*` functions.
    """
    print("NEU")
    return build_tangle_model(
        uploaded,
        agreement,
        perplexity,
        max_same_parameter=(max_same_parameter - 1),
        max_number_of_seps=max_number_of_seps,
        order_function=order_function,
        normalized_values_for_tsne=normalized_values_for_tsne,
        exclude_columns=exclude_cols,
    )

# ------------------------
# Parameter presets
# ------------------------

def build_global_parameter_set():
    """
    Snapshot the current search parameters from the session state.

    Returns a plain dict of the search-relevant widget values, which is what
    gets stored as a named preset.
    """
    return {
        "agreement": int(st.session_state.get("agreement", 10)),
        "max_number_of_seps": int(st.session_state.get("max_number_of_seps", 10)),
        "order_function": st.session_state.get("order_function", "01"),
        "perplexity": int(st.session_state.get("perplexity", 10)),
        "normalized_values_for_tsne": bool(st.session_state.get("normalized_values_for_tsne", True)),
        "max_same_parameter": int(st.session_state.get("max_same_parameter", 2)),
        "show_only_non_numeric_exclude": bool(
            st.session_state.get("show_only_non_numeric_exclude", False)
        ),
        "exclude_cols": list(st.session_state.get("exclude_cols", [])),
    }


def apply_global_parameter_set(param_set: dict):
    """
    Write a saved preset back into the session state (inverse of
    `build_global_parameter_set`).

    Because the widgets are bound to these keys, the caller must trigger a rerun
    afterwards for the new values to show up.
    """
    st.session_state["agreement"] = int(param_set.get("agreement", 10))
    st.session_state["max_number_of_seps"] = int(param_set.get("max_number_of_seps", 10))
    st.session_state["order_function"] = param_set.get("order_function", "01")
    st.session_state["perplexity"] = int(param_set.get("perplexity", 10))
    st.session_state["normalized_values_for_tsne"] = bool(
        param_set.get("normalized_values_for_tsne", True)
    )
    st.session_state["max_same_parameter"] = int(param_set.get("max_same_parameter", 2))
    st.session_state["show_only_non_numeric_exclude"] = bool(
        param_set.get("show_only_non_numeric_exclude", False)
    )
    st.session_state["exclude_cols"] = list(param_set.get("exclude_cols", []))

def vis_global_parameter_sets():
    """
    Sidebar section for saving, loading and deleting parameter presets.

    Presets are kept in `st.session_state["saved_global_parameter_sets"]` as
    {"name", "params"} entries and live only for the session. Loading, deleting
    and clearing each end in `st.rerun` so the widgets pick up the change.
    """
    st.subheader("Saved parameter sets")

    with st.form("save_global_parameter_set_form"):
        preset_name = st.text_input(
            "Preset name",
            value=f"Preset {len(st.session_state['saved_global_parameter_sets']) + 1}",
            key="global_parameter_set_name_input",
        )
        save_submitted = st.form_submit_button("Save current parameters")

    if save_submitted:
        name_clean = preset_name.strip()
        if not name_clean:
            st.warning("Please provide a preset name.")
        else:
            st.session_state["saved_global_parameter_sets"].append(
                {
                    "name": name_clean,
                    "params": build_global_parameter_set(),
                }
            )
            st.rerun()

    presets = st.session_state.get("saved_global_parameter_sets", [])

    if not presets:
        st.info("No saved parameter sets yet.")
        return

    preset_names = [p["name"] for p in presets]
    selected_name = st.selectbox(
        "Load saved parameter set",
        options=preset_names,
        key="selected_global_parameter_set_name",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Load preset", key="load_global_parameter_set_button"):
            preset = next((p for p in presets if p["name"] == selected_name), None)
            if preset is not None:
                apply_global_parameter_set(preset["params"])
                st.rerun()

    with col2:
        if st.button("Delete preset", key="delete_global_parameter_set_button"):
            st.session_state["saved_global_parameter_sets"] = [
                p for p in presets if p["name"] != selected_name
            ]
            st.rerun()

    with col3:
        if st.button("Clear all presets", key="clear_global_parameter_sets_button"):
            st.session_state["saved_global_parameter_sets"] = []
            st.rerun()

    with st.expander("Saved presets overview", expanded=False):
        overview_rows = []
        for preset in presets:
            row = {"name": preset["name"]}
            row.update(preset["params"])
            overview_rows.append(row)
        st.dataframe(pd.DataFrame(overview_rows), width="stretch")


# ------------------------
# Visualization sections
# ------------------------

def vis_tangle_membership(tangle_dict, customdata, hover_cols):
    """
    Single-tangle t-SNE view for the tangle selected in the rule-list.

    Shows the continuous score colouring by default; the checkbox switches to a
    binary membership view whose threshold is chosen with a slider.
    """
    st.subheader("Tangle Score Visualization")
    selected_idx = st.session_state["selected_tangle"]
    if selected_idx is None:
        st.info("Please select a tangle above.")
        return

    show_membership = st.checkbox(f"Show Membership of Tangle {selected_idx}", value=False)
    if show_membership:
        membership_threshold = st.slider("Select threshold for membership: ", 0.0, 1.0, 0.8, step=0.05)
        membership = membership_from_scores(tangle_dict.get("scores_all"), selected_idx, membership_threshold)
    else:
        st.session_state["threshold"] = None
        membership = None

    st.plotly_chart(
        fig_tsne_membership(tangle_dict.get("embedding"), tangle_dict.get("scores_all"), customdata, membership, selected_idx, hover_cols ),
    )

def vis_tangle_descriptions(tangle_dict):
    """
    Rule-list column: pick the active tangle and select individual rules.

    Renders the human-readable conditions from
    `core_tangles.describe_tangle_conditions`. The active tangle drives every
    other view via `st.session_state["selected_tangle"]`; the per-rule
    checkboxes feed `st.session_state["selected_rule_ids"]`, which
    `vis_rule_combination_tsne` turns into a highlighted intersection. The full
    rule-list of all tangles is additionally offered as a CSV download.
    """
    st.subheader("Tangles Rule-List")

    descriptions = tangle_dict.get("tangle_descriptions") or []
    if not descriptions:
        st.info("No tangles found.")
        return

    n_tangles = len(descriptions)
    tangle_names = [t["name"] for t in descriptions]

    current_idx = st.session_state.get("selected_tangle", 0)
    if current_idx is None or current_idx < 0 or current_idx >= n_tangles:
        current_idx = 0

    if n_tangles <= 10:
        selected_idx = st.radio(
            "Active tangle",
            options=list(range(n_tangles)),
            index=current_idx,
            format_func=lambda i: tangle_names[i],
            key="active_tangle_radio",
        )
    else:
        selected_idx = st.selectbox(
            "Active tangle",
            options=list(range(n_tangles)),
            index=current_idx,
            format_func=lambda i: tangle_names[i],
            key="active_tangle_select",
        )

    previous_idx = st.session_state.get("selected_tangle")
    if previous_idx != selected_idx:
        st.session_state["selected_rule_ids"] = []

    st.session_state["selected_tangle"] = selected_idx

    active = descriptions[selected_idx]
    rules = active.get("rules", [])

    st.markdown(f"Rules of **{active['name']} (active)**")
    st.caption("Select one or more rules. The t-SNE view highlights points that satisfy all selected rules.")

    selected_rule_ids = []

    for rule_idx, rule in enumerate(rules):
        checked = st.checkbox(
            rule["text"],
            value=rule_idx in st.session_state.get("selected_rule_ids", []),
            key=f"rule_check_t{selected_idx}_r{rule_idx}",
        )
        if checked:
            selected_rule_ids.append(rule_idx)

    st.session_state["selected_rule_ids"] = selected_rule_ids

    with st.expander("Show all tangles (full rule-list)", expanded=False):
        for idx, t in enumerate(descriptions):
            is_active = idx == selected_idx
            header = f"{t['name']} (active)" if is_active else t["name"]

            with st.expander(header, expanded=is_active):
                for cond in t["conditions"]:
                    st.write(f"- {cond}")

    rows = []
    for t_idx, t in enumerate(descriptions):
        tangle_name = t.get("name", f"Tangle {t_idx}")
        for r_idx, rule in enumerate(t.get("rules", [])):
            rows.append(
                {
                    "tangle_index": t_idx,
                    "tangle_name": tangle_name,
                    "rule_index": r_idx,
                    "sep_id": rule.get("sep_id"),
                    "orientation": rule.get("orientation"),
                    "rule_text": rule.get("text"),
                }
            )

    if rows:
        df_rules = pd.DataFrame(rows)
        csv_bytes = df_rules.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Rule-List as CSV",
            data=csv_bytes,
            file_name="tangle_rule_list.csv",
            mime="text/csv",
        )

def vis_tangle_feature_heatmap(tangle_dict):
    """
    Feature-effect heatmap of all tangles, with a slider for how many features
    to display (the most discriminative ones are chosen by `viz_plotly`).
    """
    st.subheader("Feature Heatmap of z-score")

    effects = tangle_dict.get("feature_effects")
    if effects is None or effects.empty:
        st.info("No feature effects available (check if there are numeric features).")
    else:
        max_feat_default = min(15, effects.shape[1])
        max_features = st.slider(
            "Number of features to show in the heatmap:",
            min_value=3,
            max_value=int(min(40, effects.shape[1])),
            value=int(max_feat_default),
            step=1,
        )
        heatmap_fig = fig_tangle_feature_heatmap(effects, max_features=max_features)
        st.plotly_chart(heatmap_fig, width="stretch")

def vis_parallel_coordinates(tangle_dict):
    """
    Parallel-coordinates profile of the points inside the active tangle.

    The user picks the feature axes (first six numeric columns by default) and
    the membership threshold; the number of matching points is reported so an
    empty selection is recognizable.
    """
    selected_idx = st.session_state["selected_tangle"]
    if selected_idx is None:
        st.info("Please select a tangle above.")
        return

    st.subheader("Feature profiles of selected tangle")

    df_numeric = tangle_dict.get("df")
    if df_numeric is None or df_numeric.empty:
        st.info("No numeric features available.")
        return

    default_features = list(df_numeric.columns[: min(6, len(df_numeric.columns))])

    selected_features = st.multiselect(
        "Select features for the axes of the visualization:",
        options=list(df_numeric.columns),
        default=default_features,
    )

    membership_threshold = st.slider(
        "Threshold for points in this view:",
        0.0, 1.0, 0.3, step=0.05,
        key="parallel_threshold",
    )

    membership = membership_from_scores(
        tangle_dict.get("scores_all"),
        selected_idx,
        membership_threshold,
    )

    n_selected = int(membership.sum())
    st.write(f"Points in selected tangle: {n_selected}")

    if n_selected == 0:
        st.warning("No points match the current threshold.")
        return

    fig = fig_parallel_coordinates(
        df_numeric=df_numeric,
        membership=membership,
        selected_features=selected_features,
        score_values=tangle_dict.get("scores_all")[:, selected_idx],
    )
    st.plotly_chart(fig, width="stretch")


def vis_compare_multiple_tangles(tangle_dict, customdata, hover_cols):
    """
    Small-multiples view: several tangles side by side in the same embedding.

    Tangles are chosen freely (the first four by default) and share one
    membership threshold, so the highlighted regions can be compared directly.
    """
    st.subheader("Multiplot Tangle Visualization")

    n_tangles = tangle_dict.get("scores_all").shape[1]
    options = list(range(n_tangles))

    default_selection = options[: min(4, len(options))]
    selected_tangles = st.multiselect(
        "Select tangles to compare:",
        options=options,
        default=default_selection,
    )

    if not selected_tangles:
        st.info("Please select at least one tangle.")
        return

    threshold = st.slider(
        "Threshold for highlighted membership:",
        0.0, 1.0, 0.8, step=0.05,
        key="compare_tangles_threshold",
    )

    fig = fig_tsne_compare_tangles(
        embedding=tangle_dict.get("embedding"),
        scores_all=tangle_dict.get("scores_all"),
        selected_tangles=selected_tangles,
        threshold=threshold,
        customdata=customdata,
        hover_columns=hover_cols,
        n_cols=2,
    )

    st.plotly_chart(fig, width="stretch")

def vis_compare_agreement_scans(tangle_dict, max_number_of_seps, order_function, max_same_parameter):
    """
    Agreement-scan tab: run scans over an agreement range and compare saved runs.

    The form defines the range (start / end / step, capped at 40 values) and an
    optional label. "Run and save scan" scans with the currently selected order
    function; "Run scans for all order functions" repeats the same range for
    every available order function and saves one run per function. Runs
    accumulate in `st.session_state["agreement_scan_runs"]` and are drawn as one
    line each; buttons below remove the last run or all of them.

    Every scan reuses the separation system from `tangle_dict`, so only the
    tangle search itself is repeated per agreement value.
    """
    st.subheader("Agreement Parameter Scan")
    st.caption("Save multiple agreement scans and compare them in one plot.")

    with st.form("agreement_scan_compare_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            agreement_start = st.number_input(
                "Agreement start",
                min_value=1,
                value=10,
                step=1,
                key="agreement_compare_start",
            )

        with col2:
            agreement_end = st.number_input(
                "Agreement end",
                min_value=1,
                value=50,
                step=1,
                key="agreement_compare_end",
            )

        with col3:
            agreement_step = st.number_input(
                "Agreement step",
                min_value=1,
                value=5,
                step=1,
                key="agreement_compare_step",
            )

        run_label = st.text_input(
            "Run label (optional)",
            key="agreement_compare_label",
        )

        col1_1, col2_2, col3_3 = st.columns(3)
        with col1_1:
            single_submitted = st.form_submit_button("Run and save scan (current order)")

        with col2_2:
            all_submitted = st.form_submit_button("Run scans for all order functions")

    if single_submitted or all_submitted:
        if agreement_start > agreement_end:
            st.warning("Agreement start must be smaller than or equal to agreement end.")
            return

        agreements = list(
            range(
                int(agreement_start),
                int(agreement_end) + 1,
                int(agreement_step),
            )
        )

        if len(agreements) > 40:
            st.warning(
                "Please choose a smaller range or a larger step size (max. 40 values)."
            )
            return

    if single_submitted:
        existing_labels = [r["label"] for r in st.session_state["agreement_scan_runs"]]
        label_clean = run_label.strip() or _next_run_label(existing_labels)

        if label_clean in existing_labels:
            st.warning(
                f"A scan with the label '{label_clean}' already exists. "
                "Please choose a different label."
            )
            return

        with st.spinner("Running and saving agreement scan..."):
            results = scan_agreement_range_from_tangle_dict(
                tangle_dict=tangle_dict,
                agreements=agreements,
                max_number_of_seps=max_number_of_seps,
                order_function=order_function,
            )

        st.session_state["agreement_scan_runs"].append(
            {
                "label": label_clean,
                "params": {
                    "agreement_start": int(agreement_start),
                    "agreement_end": int(agreement_end),
                    "agreement_step": int(agreement_step),
                    "max_number_of_seps": int(max_number_of_seps),
                    "order_function": order_function,
                    "max_same_parameter": max_same_parameter,
                },
                "results": results.copy(),
            }
        )

        st.rerun()
    if all_submitted:
        # one run per order function, each labelled "<base> | order=<name>"
        order_functions_to_run = ["01", "01-biased", "02", "03", "04", "cut", "radiocut"]

        existing_labels = [r["label"] for r in st.session_state["agreement_scan_runs"]]

        with st.spinner("Running and saving scans for all order functions..."):
            for of in order_functions_to_run:
                if run_label.strip():
                    base_label = run_label.strip()
                else:
                    base_label = _next_run_label(existing_labels)

                label_for_run = f"{base_label} | order={of}"

                existing_labels.append(label_for_run)

                results = scan_agreement_range_from_tangle_dict(
                    tangle_dict=tangle_dict,
                    agreements=agreements,
                    max_number_of_seps=max_number_of_seps,
                    order_function=of,
                )

                st.session_state["agreement_scan_runs"].append(
                    {
                        "label": label_for_run,
                        "params": {
                            "agreement_start": int(agreement_start),
                            "agreement_end": int(agreement_end),
                            "agreement_step": int(agreement_step),
                            "max_number_of_seps": int(max_number_of_seps),
                            "order_function": of,
                            "max_same_parameter": max_same_parameter,
                        },
                        "results": results.copy(),
                    }
                )

        st.rerun()

    runs = st.session_state.get("agreement_scan_runs", [])

    if not runs:
        st.info("No saved agreement scans yet.")
        return


    # concatenate the saved runs into one long-format table for the line plot,
    # plus a metadata table documenting each run's configuration
    rows_long = []
    rows_meta = []

    for i, run in enumerate(runs, start=1):
        label = run["label"]
        params = run["params"]
        df_run = run["results"].copy()

        df_run["run_label"] = label
        rows_long.append(df_run)

        rows_meta.append(
            {
                "run": i,
                "label": label,
                "agreement_start": params["agreement_start"],
                "agreement_end": params["agreement_end"],
                "agreement_step": params["agreement_step"],
                "max_number_of_seps": params["max_number_of_seps"],
                "order_function": params["order_function"],
                "max_same_parameter": params["max_same_parameter"],
            }
        )

    results_long = pd.concat(rows_long, ignore_index=True)
    meta_df = pd.DataFrame(rows_meta)

    st.plotly_chart(fig_agreement_scan_comparison(results_long), width="stretch")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Remove last saved scan", key="remove_last_agreement_scan"):
            st.session_state["agreement_scan_runs"] = runs[:-1]
            st.rerun()

    with col_btn2:
        if st.button("Clear all saved scans", key="clear_all_agreement_scans"):
            st.session_state["agreement_scan_runs"] = []
            st.rerun()

    with st.expander("Saved scan configurations", expanded=False):
        st.dataframe(meta_df, width="stretch")

def vis_rule_combination_tsne(tangle_dict, customdata, hover_cols):
    """
    Highlight the points satisfying all rules selected in the rule-list.

    The mask is built directly from the separation matrix S: a point belongs to
    the intersection if its entry in every selected separation column matches
    that rule's orientation. This mirrors the definition of the tangle's
    conditions exactly, so the highlighted region is the literal intersection of
    the chosen oriented separations rather than a score-based approximation.
    """
    st.subheader("Selected Rules Visualization")

    selected_idx = st.session_state.get("selected_tangle")
    if selected_idx is None:
        st.info("Please select a tangle in the rule-list.")
        return

    descriptions = tangle_dict.get("tangle_descriptions") or []
    if selected_idx < 0 or selected_idx >= len(descriptions):
        st.info("Invalid selected tangle.")
        return

    active = descriptions[selected_idx]
    rules = active.get("rules", [])
    selected_rule_ids = st.session_state.get("selected_rule_ids", [])

    if not selected_rule_ids:
        st.info("Select one or more rules in the rule-list to highlight their intersection.")
        return

    S = tangle_dict.get("S")
    if S is None:
        st.info("No separation system available.")
        return

    # start with all points and intersect one selected rule at a time
    membership = np.ones(S.shape[0], dtype=bool)
    selected_rule_texts = []

    for rule_idx in selected_rule_ids:
        if rule_idx < 0 or rule_idx >= len(rules):
            continue
        rule = rules[rule_idx]
        sep_id = rule["sep_id"]
        orientation = rule["orientation"]

        membership &= (S[:, sep_id] == orientation)
        selected_rule_texts.append(rule["text"])

    n_selected = int(membership.sum())
    st.write(f"Points satisfying all selected rules: {n_selected}")

    with st.expander("Selected rules", expanded=False):
        for txt in selected_rule_texts:
            st.write(f"- {txt}")

    title = f"t-SNE: intersection of selected rules in {active['name']}"
    fig = fig_tsne_rule_combination(
        embedding=tangle_dict.get("embedding"),
        membership=membership,
        customdata=customdata,
        hover_columns=hover_cols,
        title=title,
    )
    st.plotly_chart(fig, width="stretch")

def vis_run_summary(
    tangle_dict,
    agreement,
    order_function,
    use_order_function,
    max_number_of_seps,
    normalized_values_for_tsne,
    exclude_cols,
):
    """
    Textual summary of the current run shown at the top of the overview tab.

    Reports the size of the analyzed data, how many numeric and categorical
    separations were built, how many maximal tangles the search returned, and
    which settings produced them — so a screenshot of the plots stays
    interpretable on its own.
    """
    st.subheader("Run summary")

    n_points = int(len(tangle_dict.get("df")))
    n_numeric_features = int(tangle_dict.get("df").shape[1])
    n_all_columns = int(tangle_dict.get("df_all_columns").shape[1])
    n_separations = int(tangle_dict.get("S").shape[1]) if tangle_dict.get("S") is not None else 0
    n_tangles = int(tangle_dict.get("scores_all").shape[1]) if tangle_dict.get("scores_all") is not None else 0

    sep_meta = tangle_dict.get("sep_meta") or []
    n_numeric_seps = sum(1 for s in sep_meta if getattr(s, "kind", None) == "numeric")
    n_categorical_seps = sum(1 for s in sep_meta if getattr(s, "kind", None) == "categorical")

    if use_order_function:
        order_text = f"using order function {order_function}"
    else:
        order_text = f"without an order function and with a maximum of {int(max_number_of_seps)} separations"

    normalization_text = "normalized" if normalized_values_for_tsne else "non-normalized"

    excluded_text = (
        f"{len(exclude_cols)} excluded columns"
        if exclude_cols
        else "no excluded columns"
    )

    st.info(
        f"""
For this run, the prototype analyzed **{n_points} data points** with **{n_numeric_features} numeric features** out of **{n_all_columns} total columns**.
It constructed **{n_separations} candidate separations** in total, including **{n_numeric_seps} numeric** and **{n_categorical_seps} categorical** separations.
With **agreement {agreement}**, the search found **{n_tangles} maximal tangles**, {order_text}.
The t-SNE embedding was computed on **{normalization_text}** feature values, with **{excluded_text}** kept out of the tangle search.
        """
    )

# ------------------------
# Page layout
# ------------------------

def main():
    """
    Build the page: search setup, visualization tabs and rule-list.

    Nothing is rendered until a CSV is uploaded. The uploaded file is first read
    once for a preview (to offer its columns as exclude/hover options and to
    derive the default agreement), then rewound with `seek(0)` so the pipeline
    can read it again.
    """
    st.set_page_config(page_title="Tangles Explorer", layout="wide")
    st.title("Tangles Explorer (Prototype)")

    col1, col2, col3 = st.columns([1, 5, 1])

    with col1:
        st.subheader("Search setup")

        uploaded = st.file_uploader("Upload CSV-Dataset", type=["csv"])

        if uploaded is None:
            st.info("Please upload a CSV-Dataset")
            return

        df_preview = pd.read_csv(uploaded)
        numeric_cols = list(df_preview.select_dtypes(include=["number"]).columns)
        non_numeric_cols = [c for c in df_preview.columns if c not in numeric_cols]

        default_agreement = min(round(len(df_preview.index) / 9), 100)

        _init_session_state(default_agreement)
        st.divider()

        st.subheader("Core parameters")
        st.caption("The agreement parameter and the order function are the main controls for the tangle search.")

        agreement = st.number_input(
            "Agreement parameter n",
            min_value=1,
            step=1,
            key="agreement",
        )

        use_order_function = st.checkbox(
            "Use order function",
            key="use_order_function",
            value=True,
        )

        if use_order_function:
            order_function = st.selectbox(
                "Order function",
                ("01", "01-biased", "02", "03", "04", "cut", "radiocut"),
                key="order_function",
            )
            max_number_of_seps = int(st.session_state.get("max_number_of_seps", 10))
        else:
            order_function = None
            max_number_of_seps = st.number_input(
                "Maximum number of separations in tangle",
                min_value=1,
                step=1,
                key="max_number_of_seps",
            )

        st.divider()

        with st.expander("Advanced settings", expanded=False):
            perplexity = st.number_input(
                "t-SNE Perplexity",
                min_value=1,
                key="perplexity",
            )

            normalized_values_for_tsne = st.checkbox(
                "Normalize values for t-SNE",
                key="normalized_values_for_tsne",
            )

            max_same_parameter = st.number_input(
                "Maximal number of same feature partitions per tangle (numerical features)",
                min_value=2,
                key="max_same_parameter",
            )

        st.divider()

        with st.expander("Columns to exclude from search", expanded=False):
            show_only_non_numeric_exclude = st.checkbox(
                "Show only non-numeric columns",
                key="show_only_non_numeric_exclude",
            )

            exclude_options = (
                non_numeric_cols
                if show_only_non_numeric_exclude
                else list(df_preview.columns)
            )

            exclude_cols = st.multiselect(
                "Columns to exclude from Tangle search:",
                options=exclude_options,
                key="exclude_cols",
            )

        uploaded.seek(0)
        tangle_dict = run(
            uploaded,
            agreement,
            max_same_parameter,
            perplexity,
            int(max_number_of_seps),
            order_function,
            normalized_values_for_tsne,
            exclude_cols,
        )

        st.divider()

        with st.expander("Customize hover options", expanded=False):
            show_only_numeric_hover = st.checkbox("Show only numeric columns", key="show_only_numeric_hover")

            if show_only_numeric_hover:
                hover_cols = st.multiselect(
                    label="Select the columns to display in the hover tooltip:",
                    options=list(tangle_dict.get("df").columns),
                )
            else:
                hover_cols = st.multiselect(
                    label="Select the columns to display in the hover tooltip:",
                    options=list(tangle_dict.get("df_all_columns").columns),
                )

        st.divider()

        vis_global_parameter_sets()

        customdata = build_customdata(tangle_dict.get("df_all_columns"), hover_cols)
    with col3:
        vis_tangle_descriptions(tangle_dict)
    with col2:
        st.subheader("Visualizations")

        tab_overview, tab_membership, tab_features, tab_view_partitions, tab_agreement = st.tabs(
            ["Overview", "Membership", "Features", "View partitions", "Agreement scans"]
        )

        with tab_overview:
            vis_run_summary(tangle_dict, agreement, order_function, use_order_function, max_number_of_seps, normalized_values_for_tsne, exclude_cols)

            st.subheader("Tree of Tangles")
            tree_fig = fig_tree_of_tangles(
                tangle_dict.get("embedding"),
                tangle_dict.get("tangles"),
                agreement=agreement,
            )
            st.pyplot(tree_fig, clear_figure=True)

        with tab_membership:
            vis_tangle_membership(tangle_dict, customdata, hover_cols)
            vis_compare_multiple_tangles(tangle_dict, customdata, hover_cols)

        with tab_features:
            vis_tangle_feature_heatmap(tangle_dict)
            vis_parallel_coordinates(tangle_dict)

        with tab_view_partitions:
            vis_rule_combination_tsne(tangle_dict, customdata, hover_cols)

        with tab_agreement:
            vis_compare_agreement_scans(tangle_dict, max_number_of_seps, order_function, max_same_parameter)

main()