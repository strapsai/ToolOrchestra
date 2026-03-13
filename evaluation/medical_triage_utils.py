import json
import re

DAY_LABELS = [
    # Dead
    "clearly_see_lack_of_chest_or_abdominal_movement_with_no_other_motion",
    "death_assessed_for_casualty",
    "death_not_assessable",
    # Severe Hemorrhage
    "active_bleeding_observed",
    "blood_pooling_observed",
    "extensive_blood_on_body",
    "hemorrhage_assessed_absent",
    "hemorrhage_not_assessable",
    # Respiratory
    "tripod_position_observed",
    "gasping_or_labored_breathing",
    "abnormal_head_neck_posture",
    "cyanosis_observed",
    "respiratory_distress_assessed_absent",
    "respiratory_distress_not_assessable",
    # Trauma - head/neck
    "wound_head_neck_observed",
    "head_neck_assessed_no_wound",
    "head_neck_not_assessable",
    # Trauma - torso
    "wound_torso_observed",
    "torso_assessed_no_wound",
    "torso_not_assessable",
    # Trauma - upper extremities
    "wound_upper_extremity_observed",
    "amputation_upper_extremity_observed",
    "upper_extremities_assessed_no_wound",
    "upper_extremities_not_assessable",
    # Trauma - lower extremities
    "wound_lower_extremity_observed",
    "amputation_lower_extremity_observed",
    "lower_extremities_assessed_no_wound",
    "lower_extremities_not_assessable",
    # Alertness - ocular
    "eyes_open_with_blinking",
    "eyes_open_no_blinking",
    "eyes_closed_throughout",
    "eyes_not_assessable",
    # Alertness - verbal
    "verbal_response_normal",
    "verbal_response_abnormal",
    "verbal_response_absent",
    "verbal_not_assessable",
    # Alertness - motor/posture
    "ambulatory",
    "standing_unsupported",
    "sitting_unsupported",
    "sitting_supported",
    "lying_down",
    "posture_not_assessable",
    # Alertness - motor/movement
    "coordinated_movement",
    "minimal_movement_only",
    "no_movement",
    "movement_not_assessable",
    # Physiological
    "disoriented_movement_observed",
    "protective_hand_placement_observed",
    # Treatment
    "tourniquet_applied_observed",
    "bandaging_applied_observed",
    "medic_actively_treating",
    "treatment_assessed_absent",
    "treatment_not_assessable",
    # Subject type
    "subject_is_casualty_human_actor",
    "subject_is_casualty_manikin",
    "subject_is_medic",
    "subject_is_robot",
    # Context
    "close_inspection_achieved",
    "video_quality_insufficient",
]

NIGHT_LABELS = [
    # Dead
    "thermal_no_chest_abdominal_movement_with_no_other_motion",
    "thermal_death_assessed_for_casualty",
    "thermal_death_not_assessable",
    # Severe Hemorrhage
    "thermal_active_bleeding_observed",
    "thermal_blood_pooling_observed",
    "thermal_extensive_blood_on_body",
    "thermal_hemorrhage_assessed_absent",
    "thermal_hemorrhage_not_assessable",
    # Respiratory
    "thermal_tripod_position_observed",
    "thermal_gasping_or_labored_breathing",
    "thermal_abnormal_head_neck_posture",
    "thermal_respiratory_distress_assessed_absent",
    "thermal_respiratory_distress_not_assessable",
    # Trauma - head/neck
    "thermal_wound_head_neck_observed",
    "thermal_head_neck_assessed_no_wound",
    "thermal_head_neck_not_assessable",
    # Trauma - torso
    "thermal_wound_torso_observed",
    "thermal_torso_assessed_no_wound",
    "thermal_torso_not_assessable",
    # Trauma - upper extremities
    "thermal_wound_upper_extremity_observed",
    "thermal_amputation_upper_extremity_observed",
    "thermal_upper_extremities_assessed_no_wound",
    "thermal_upper_extremities_not_assessable",
    # Trauma - lower extremities
    "thermal_wound_lower_extremity_observed",
    "thermal_amputation_lower_extremity_observed",
    "thermal_lower_extremities_assessed_no_wound",
    "thermal_lower_extremities_not_assessable",
    # Alertness - ocular
    "thermal_eyes_open",
    "thermal_eyes_closed",
    "thermal_eyes_not_assessable",
    # Alertness - verbal
    "thermal_verbal_response_detected",
    "thermal_verbal_response_absent",
    "thermal_verbal_not_assessable",
    # Alertness - motor/posture
    "thermal_ambulatory",
    "thermal_standing_unsupported",
    "thermal_sitting_unsupported",
    "thermal_sitting_supported",
    "thermal_lying_down",
    "thermal_posture_not_assessable",
    # Alertness - motor/movement
    "thermal_coordinated_movement",
    "thermal_minimal_movement",
    "thermal_no_movement",
    "thermal_movement_not_assessable",
    # Physiological
    "thermal_disoriented_movement_observed",
    "thermal_protective_hand_placement_observed",
    # Treatment
    "thermal_tourniquet_applied_observed",
    "thermal_bandaging_applied_observed",
    "thermal_medic_actively_treating",
    "thermal_treatment_assessed_absent",
    "thermal_treatment_not_assessable",
    # Subject type
    "thermal_subject_is_casualty",
    "thermal_subject_is_medic",
    "thermal_subject_is_robot",
    # Context
    "thermal_close_inspection_achieved",
    "thermal_image_quality_insufficient",
]

# Temporal labels — subset requiring video/temporal input
TEMPORAL_LABELS = [
    "clearly_see_lack_of_chest_or_abdominal_movement_with_no_other_motion",
    "gasping_or_labored_breathing",
    "eyes_open_with_blinking",
    "eyes_open_no_blinking",
    "eyes_closed_throughout",
    "coordinated_movement",
    "minimal_movement_only",
    "no_movement",
    "disoriented_movement_observed",
    "ambulatory",
    # Night/thermal equivalents
    "thermal_no_chest_abdominal_movement_with_no_other_motion",
    "thermal_gasping_or_labored_breathing",
    "thermal_eyes_open",
    "thermal_eyes_closed",
    "thermal_coordinated_movement",
    "thermal_minimal_movement",
    "thermal_no_movement",
    "thermal_disoriented_movement_observed",
    "thermal_ambulatory",
]

## Helpers

def classify_labels(label_keys, temporal_labels):
    temporal_set = set(temporal_labels)
    static_keys = [k for k in label_keys if k not in temporal_set]
    temporal_keys = [k for k in label_keys if k in temporal_set]
    return static_keys, temporal_keys


def select_label_set(sensor_name):
    if sensor_name == "ir":
        return "night"
    return "day"


def get_label_keys(sensor_name):
    if sensor_name == "ir":
        return list(NIGHT_LABELS)
    return list(DAY_LABELS)


## Prompt Construction:

def build_static_prompt(label_keys):
    labels_block = ", ".join(f'"{k}"' for k in label_keys)
    return f"""You are assessing a casualty in a mass casualty incident.
Analyze the provided image(s) and produce a structured triage assessment.

Return a JSON object with EXACTLY these keys, each set to
1 (observed/true) or 0 (not observed/false):

[{labels_block}]

Rules:
- Every key must be present in your response.
- Values must be integer 0 or 1 (not booleans, not strings).
- If you cannot assess a category, set the corresponding
  "*_not_assessable" key to 1 and related observables to 0.
- Return ONLY the JSON object. No markdown fences, no explanation."""

def build_temporal_prompt(label_keys):
    labels_block = ", ".join(f'"{k}"' for k in label_keys)
    return f"""You are assessing a casualty in a mass casualty incident.
Analyze the provided video sequence for movement, breathing, and
temporal patterns. Focus on changes across frames.

Return a JSON object with EXACTLY these keys, each set to
1 (observed/true) or 0 (not observed/false):

[{labels_block}]

Rules:
- Every key must be present in your response.
- Values must be integer 0 or 1 (not booleans, not strings).
- Pay attention to motion between frames for movement labels.
- For breathing assessment, look for chest/abdominal movement patterns.
- If you cannot assess a category, set the corresponding
  "*_not_assessable" key to 1 and related observables to 0.
- Return ONLY the JSON object. No markdown fences, no explanation."""

## Response Parsing (Taken from Varun's Repo)

def parse_vlm_response(content):
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract JSON from VLM response")

def validate_labels(data, expected_keys):
    errors = []
    data_keys = set(data.keys())
    expected = set(expected_keys)

    missing = expected - data_keys
    if missing:
        errors.append(f"Missing keys: {sorted(missing)}")

    extra = data_keys - expected
    if extra:
        errors.append(f"Unexpected keys: {sorted(extra)}")

    for k, v in data.items():
        if v not in (0, 1):
            errors.append(f"Bad value for '{k}': {v} (must be 0 or 1)")

    return errors