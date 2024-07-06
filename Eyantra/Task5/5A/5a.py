import networkx as nx
import esp32_wifi_communication
import cv2
import subprocess

G = nx.Graph()
G.add_nodes_from(range(12))
G.add_edges_from([
    (0,1),
    (1,2),(1,3),
    (2,4),(2,5),
    (3,6),(3,4),
    (4,7),(4,5),
    (5,8),
    (6,9),(6,7),
    (7,10),(7,8),
    (8,11),
    (9,10),(9,11),
    (10,11)
])

source_vertex = 0

def convert_and_relay_path(events):
    # This is a placeholder example, replace it with your actual implementation
    machine_code = " ".join(events)  # Convert path to space-separated string
    esp32_wifi_communication.send_machine_code(machine_code)

def detect_events(frame):
    
    detected_events = []
    with open("detected_events.txt", 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        node, event = parts
                        detected_events.append((node.strip(), event.strip().lower()))

    return detected_events

event_priority_order = ["fire", "destroyed buildings", "humanitarian aid and rehabilitation", "military vehicles", "combat"]
event_priority_counter = {event: 0 for event in event_priority_order}

if __name__ == "__main__":

    # input("Press Enter to start the next run...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        exit(1)

    cap.release()
    cv2.destroyAllWindows()
    subprocess.Popen(['python', 'geolocation.py'])
    detected_events = detect_events(frame)

    sorted_events = sorted(detected_events, key=lambda x: event_priority_order.index(x[1]), reverse=True)

    # Initialize variables
    current_vertex = source_vertex
    full_path = []

    # Iterate through sorted events
    for event in sorted_events:
        # Calculate shortest path from current vertex to the event
        path_to_event = nx.shortest_path(G, source=current_vertex, target=event)
        
        # Update current vertex
        current_vertex = event
        
        # Append the path to the full path
        full_path += path_to_event

    # Calculate the final path to the source vertex
    path_to_source = nx.shortest_path(G, source=current_vertex, target=source_vertex)

    # Append the path to the full path
    full_path += path_to_source

    # Convert and relay the full path
    convert_and_relay_path(full_path)

    # Update event priority counter for all detected events
    for event in sorted_events:
        event_priority_counter[event] += 1


    # Print or use the updated EP counter as needed
    print("Event Priority Counter:", event_priority_counter)

    # Sort the event_priority_order based on the count
    # event_priority_order = sorted(event_priority_order, key=lambda x: event_priority_counter[x], reverse=True)
    # print("Sorted Event Priority Order:", event_priority_order)
