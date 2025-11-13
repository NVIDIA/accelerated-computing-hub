# How to Create a Brev Launchable from Accelerated Computing Hub Content

## Instructions

- Get the GitHub web URL to the Docker Compose file for the Accelerated Computing Hub content from [the README](../README.md).
- Go to [brev.nvidia.com](https://brev.nvidia.com).
- Login and select correct organization.
- Click "Launchables".

### Code Page

- Click "I have code files in a git repository".
- Select "Enter a public repository, notebook file, or markdown file URL" textbox and input "https://github.com/NVIDIA/accelerated-computing-hub".
- Click "With container(s)".
- Click "Next".

### Container Page

- Click "Docker Compose".
- Click "I Have an existing docker-compose.yaml file".
- Click "Provide GitHub/Gitlab URL".
- Select "GitHub/GitLab URL" textbox and input the content's Docker Compose file.
- Click "Validate".
- Click "Next".

### Services Page

- Click "No, I don't want Jupyter".
- Scroll to "Secure Links".
- Rename "tunnel-4" to "jupyter" on port 8888.
- Rename "tunnel-5" to "jupyter" on port 8080.
- Delete "tunnel-6".
- Click "TCP/UDP Ports".
- Select "Port or port range" textbox and input "3478".
- Click "Add Rule".
- Click "Next".

### Compute Page

- Select an instance type and provider that meets the requirements for the content from [the README](../README.md)
- Select "Data Storage" textbox and enter amount.
- Click "Next".

### Review Page

- Select "Enter a name for your Launchable" textbox and input the content's name.
- Click "Create Launchable".
- Click "View Deploy Page".

## Screenshots

![Screenshot 24](images/brev_create_launchable__24.png)
![Screenshot 25](images/brev_create_launchable__25.png)
![Screenshot 26](images/brev_create_launchable__26.png)

### Code Page

![Screenshot 27](images/brev_create_launchable__27.png)
![Screenshot 28](images/brev_create_launchable__28.png)
![Screenshot 29](images/brev_create_launchable__29.png)
![Screenshot 30](images/brev_create_launchable__30.png)

### Container Page

![Screenshot 31](images/brev_create_launchable__31.png)
![Screenshot 32](images/brev_create_launchable__32.png)
![Screenshot 33](images/brev_create_launchable__33.png)
![Screenshot 34](images/brev_create_launchable__34.png)
![Screenshot 35](images/brev_create_launchable__35.png)
![Screenshot 36](images/brev_create_launchable__36.png)
![Screenshot 37](images/brev_create_launchable__37.png)

### Services Page

![Screenshot 38](images/brev_create_launchable__38.png)
![Screenshot 39](images/brev_create_launchable__39.png)
![Screenshot 40](images/brev_create_launchable__40.png)
![Screenshot 41](images/brev_create_launchable__41.png)
![Screenshot 42](images/brev_create_launchable__42.png)
![Screenshot 43](images/brev_create_launchable__43.png)
![Screenshot 44](images/brev_create_launchable__44.png)
![Screenshot 45](images/brev_create_launchable__45.png)
![Screenshot 46](images/brev_create_launchable__46.png)
![Screenshot 47](images/brev_create_launchable__47.png)

### Compute Page

![Screenshot 48](images/brev_create_launchable__48.png)
![Screenshot 49](images/brev_create_launchable__49.png)
![Screenshot 50](images/brev_create_launchable__50.png)

### Review Page

![Screenshot 51](images/brev_create_launchable__51.png)
![Screenshot 52](images/brev_create_launchable__52.png)
![Screenshot 53](images/brev_create_launchable__53.png)
