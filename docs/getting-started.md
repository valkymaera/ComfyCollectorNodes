# Getting Started

## Installation

=== "ComfyUI-Manager"

    1. In ComfyUI, open **Manager** and choose **Custom Nodes Manager**.
    2. Search for **ComfyCollectorNodes** and click **Install**.
    3. Restart ComfyUI when prompted.

    If the pack doesn't show up in search, use **Manager → Install via Git URL**
    instead and paste:

    ```text
    https://github.com/valkymaera/ComfyCollectorNodes
    ```

=== "Git"

    Clone the repository into your `custom_nodes` folder and restart ComfyUI:

    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/valkymaera/ComfyCollectorNodes.git
    ```

    On the portable Windows build, `custom_nodes` lives at
    `ComfyUI_windows_portable\ComfyUI\custom_nodes`.

    The pack has no extra Python dependencies, so there is no `pip install`
    step — cloning and restarting is the whole install.

After restarting, the nodes are available in the node menu — every node in the
pack carries a `(CCN)` suffix, so searching for `CCN` in the node picker lists
all of them.

## Workflows

_TODO: example workflows will be documented here — chaining the signature nodes
(Video Scrubber → Cropped Image → Image Inset), curve-scheduled CFG with the
Curve CFG Guider, and Prompt Store setups that accumulate across runs._
