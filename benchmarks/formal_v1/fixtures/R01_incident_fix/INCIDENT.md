# Incident
Requests time out because the service uses milliseconds as seconds.
Acceptance: convert timeout_ms to seconds before passing it to the client.
