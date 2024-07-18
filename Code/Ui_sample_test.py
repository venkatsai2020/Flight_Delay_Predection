from locust import HttpUser, task, between
import random

class UserBehavior(HttpUser):
    wait_time = between(1, 3)  # Time between task execution
    failure_count = 0  # Initialize a failure count attribute
    
    @task
    def predict_delay(self):
        # Load the Streamlit page
        response = self.client.get("http://localhost:8501")
        
        if response.status_code == 200:
            # Select random values for simulation (adjust as per your input options)
            depart_options = ["Kolkata (CCU)", "Bengaluru (BLR)", "Delhi (DEL)"]
            arrival_options = ["Agartala Airport", "Goa Manohar International Airport", "Jabalpur Airport"]
            airline_options = ["IndiGo", "Star Air", "Air India"]
            depart = random.choice(depart_options)
            arrival = random.choice(arrival_options)
            airline = random.choice(airline_options)
            dep_date = "2024-07-07"  # Example date format (YYYY-MM-DD)
            dep_time = "07:35"    # Example time format (HH:MM)
            flight_number = "6E7418"  # Example flight number
            
            # Simulate form submission by sending a POST request with form data
            form_data = {
                "Departing Airport": depart,
                "Arrival Airport": arrival,
                "Airline": airline,
                "Departing Date": dep_date,
                "Departing Time": dep_time,
                "Flight Number": flight_number
            }
            
            # Send the POST request to simulate form submission
            submit_response = self.client.post("http://localhost:8501", data=form_data)
            
            # Check if the form submission was successful
            if submit_response.status_code == 200 and "Predicted Delay" in submit_response.text:
                self.environment.runner.stats.log_success("predict_delay", submit_response.elapsed.total_seconds(), len(submit_response.content))
            else:
                self.failure_count += 1  # Increment failure count
        else:
            self.failure_count += 1  # Increment failure count
    
    def on_stop(self):
        # Log the total number of failures at the end of the test
        self.environment.runner.stats.log_failure("predict_delay", 0, self.failure_count)
