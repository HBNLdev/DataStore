from db.collection import Subjects, Sessions, Followups, Neuropsych, Questionnaires, SSAGAs, Internalizing

subjects_coll = Subjects()
subjects_coll.update_from_followups()

session_coll = Sessions()
session_coll.update_from_followups()

followup_coll = Followups()
followup_coll.update_from_sessions()

npsych_coll = Neuropsych()
npsych_coll.update_from_sfups()

ssaga_coll = SSAGAs()
ssaga_coll.update_from_sessions()

quest_coll = Questionnaires()
quest_coll.update_from_sessions()

int_coll = Internalizing()
int_coll.update_from_ssaga()
