x, 
        onselect,
        useblit=True,
        button=[1],              # Left mouse button only
        minspanx=5, minspany=5,  # Ignore accidental tiny clicks
        props=dict(edgecolor='red', facecolor='red', alpha=0.2, fill=True),
        interactive=True         # Keeps box active to drag edges/corners
    )