# run_app.py
import os
import sys
from streamlit.web import cli as stcli  # Streamlit-CLI programmgesteuert aufrufen[web:80]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "app.py")

    # Simuliere: `streamlit run app.py`
    sys.argv = ["streamlit", "run", script]
    sys.exit(stcli.main())