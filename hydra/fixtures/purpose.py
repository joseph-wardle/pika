# Swaps the purposes: draft must arrive and final must leave.
stage.GetAttributeAtPath("/world/draft.purpose").Set("render")
stage.GetAttributeAtPath("/world/final.purpose").Set("proxy")
