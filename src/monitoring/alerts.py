
def check_drift_status(drift_detected):
    if drift_detected:
        print("WARNING: Data drift detected!")
    else:
        print("Monitoring status: No drift detected.")
