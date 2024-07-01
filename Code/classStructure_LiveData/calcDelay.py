from datetime import datetime

class Delay:
    def __init__(self, scheduled_arrival, actual_arrival):
        self.scheduled_arrival = scheduled_arrival
        self.actual_arrival = actual_arrival
        self.delays = []

    def time_to_minutes(self, time_str):
        if isinstance(time_str, str):
            try:
                time_obj = datetime.strptime(time_str, "%I:%M %p")
            except ValueError:
                try:
                    time_obj = datetime.strptime(time_str, "%H:%M")
                except ValueError:
                    return None
            return time_obj.hour * 60 + time_obj.minute
        else:
            return None

    def calculate_delays(self):
        for scheduled, actual in zip(self.scheduled_arrival, self.actual_arrival):
            scheduled_minutes = self.time_to_minutes(scheduled)
            actual_minutes = self.time_to_minutes(actual)

            if scheduled_minutes is not None and actual_minutes is not None:
                delay_minutes = actual_minutes - scheduled_minutes
                self.delays.append(delay_minutes)
            else:
                self.delays.append(None)

    def get_delays(self):
        return self.delays
