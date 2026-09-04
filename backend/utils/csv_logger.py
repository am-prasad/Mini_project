import csv
import logging
import os

logger = logging.getLogger(__name__)


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_CSV   = os.path.join(_BASE_DIR, "data", "predictions_log.csv")

_COLUMNS = [
    "Q1", "Q2", "Q3", "Q4", "Q5",
    "Q6", "Q7", "Q8", "Q9", "Q10",
    "CONTROL", "OWNERSHIP", "REACH", "ENDURANCE",
    "AQ",
    "Target_Category",
]

_CATEGORY_MAP = {"Low": 0, "Medium": 1, "High": 2}


def _calculate_core(q: dict) -> dict:
    control   = (q["Q5"] + q["Q8"] + q["Q9"]) / 3
    ownership = (q["Q2"] + q["Q6"] + q["Q10"]) / 3
    reach     = (q["Q3"] + q["Q7"]) / 2
    endurance = (q["Q1"] + q["Q4"]) / 2
    aq        = (control + ownership + reach + endurance) / 4
    return {
        "CONTROL":   round(control,   10),
        "OWNERSHIP": round(ownership, 10),
        "REACH":     round(reach,     10),
        "ENDURANCE": round(endurance, 10),
        "AQ":        round(aq,        10),
    }


def _ensure_header():
  
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_COLUMNS).writeheader()
        logger.info(f"CSV log created from scratch: {LOG_CSV}")
    
def log_prediction(questionnaire_input: dict, aq_category: str) -> None:
    
    try:
        _ensure_header()

        q    = questionnaire_input
        core = _calculate_core(q)

        row = {
            # raw inputs
            "Q1":  q["Q1"],  "Q2":  q["Q2"],  "Q3":  q["Q3"],
            "Q4":  q["Q4"],  "Q5":  q["Q5"],  "Q6":  q["Q6"],
            "Q7":  q["Q7"],  "Q8":  q["Q8"],  "Q9":  q["Q9"],
            "Q10": q["Q10"],
          
            "CONTROL":   core["CONTROL"],
            "OWNERSHIP": core["OWNERSHIP"],
            "REACH":     core["REACH"],
            "ENDURANCE": core["ENDURANCE"],
            "AQ":        core["AQ"],
          
            "Target_Category": _CATEGORY_MAP.get(aq_category, -1),
        }

        with open(LOG_CSV, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_COLUMNS).writerow(row)

        logger.info(
            f"Prediction logged → category={aq_category} "
            f"({_CATEGORY_MAP.get(aq_category, '?')})  AQ={core['AQ']:.4f}"
        )

    except Exception as exc:
        logger.error(f"csv_logger: failed to write row — {exc}", exc_info=True)
