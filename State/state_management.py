class TaskStateManager:

    def __init__(self, targets):
        """
        targets = list of 3 table coordinates (tx, ty)
        """

        self.targets = {
            "cup": targets[0],
            "bottle": targets[1],
            "pencil": targets[2]
        }

        self.current_state = "PLACE_CUP"

        self.placed = {
            "cup": False,
            "bottle": False,
            "pencil": False
        }

    # Update state every frame
    def update(self, detected_objects):

        if self.current_state == "PLACE_CUP":
            if self.is_in_target("cup", detected_objects):
                self.placed["cup"] = True
                self.current_state = "PLACE_BOTTLE"

        elif self.current_state == "PLACE_BOTTLE":
            if self.is_in_target("bottle", detected_objects):
                self.placed["bottle"] = True
                self.current_state = "PLACE_PENCIL"

        elif self.current_state == "PLACE_PENCIL":
            if self.is_in_target("pencil", detected_objects):
                self.placed["pencil"] = True
                self.current_state = "COMPLETE"

    # Check if object is close enough to its target
    def is_in_target(self, obj_name, detected_objects):

        obj_data = detected_objects.get(obj_name)

        if obj_data is None:
            return False

        tx_obj, ty_obj = obj_data["table_coords"]
        tx_target, ty_target = self.targets[obj_name]

        threshold = 0.08  # 8% of table width

        return (
            abs(tx_obj - tx_target) < threshold and
            abs(ty_obj - ty_target) < threshold
        )

    # Which object should be guided right now?
    def get_current_object(self):

        if self.current_state == "PLACE_CUP":
            return "cup"

        if self.current_state == "PLACE_BOTTLE":
            return "bottle"

        if self.current_state == "PLACE_PENCIL":
            return "pencil"

        return None

    def is_complete(self):
        return self.current_state == "COMPLETE"

    def get_current_target(self):
        obj = self.get_current_object()
        if obj is None:
            return None
        return self.targets[obj]
