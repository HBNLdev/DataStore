from db.collection import Subjects, Sessions, Followups, Neuropsych, Questionnaires, SSAGAs, Internalizing

subjects_coll = Subjects()
subjects_coll.reset_update()
subjects_coll.update_from_followups()

session_coll = Sessions()
session_coll.reset_update()
session_coll.update_from_followups()

followup_coll = Followups()
followup_coll.reset_update()
followup_coll.update_from_sessions()

npsych_coll = Neuropsych()
npsych_coll.reset_update()
npsych_coll.update_from_sfups()

ssaga_coll = SSAGAs()
ssaga_coll.reset_update()
ssaga_coll.update_from_sessions()

quest_coll = Questionnaires()
quest_coll.reset_update()
quest_coll.update_from_sessions()

int_coll = Internalizing()
int_coll.reset_update()
int_coll.update_from_ssaga()
