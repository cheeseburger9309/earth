# Earth Texture Guide

## Quick Setup (Recommended)

1. **Download a texture:**
   - Go to: https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg
   - Right-click → "Save Image As..."
   - Save as `earth.jpg`

2. **Place in project:**
   - Move `earth.jpg` to the `public/` folder
   - The app will automatically load it!

## Best Texture Sources

### 1. NASA Visible Earth (Best Quality)
- **Website:** https://visibleearth.nasa.gov/
- **Search for:** "Blue Marble" or "Earth"
- **Recommended:** "Blue Marble: Next Generation" (8K resolution)
- **Direct download:** https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg

### 2. Solar System Scope
- **Website:** https://www.solarsystemscope.com/textures/
- **Free textures** optimized for 3D globes
- Includes day/night maps

### 3. Quick Download Script

You can also download directly using this command (run in terminal from project root):

```bash
# Download NASA Blue Marble texture (8K - large file ~15MB)
curl -o public/earth.jpg "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg"

# Or download a smaller version (2K - faster to load)
curl -o public/earth.jpg "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73884/world.topo.200401.3x2160x1080.jpg"
```

## Texture Requirements

- **Format:** JPG or PNG
- **Recommended size:** 2048x1024 to 8192x4096 pixels
- **Aspect ratio:** 2:1 (width:height) for equirectangular projection
- **File size:** Keep under 20MB for web performance

## Current Setup

The app will try to load textures in this order:
1. `/earth.jpg` (local file - fastest)
2. `/earth.png` (local file)
3. Online NASA sources (fallback)

## Tips

- **For best performance:** Use a local file (2048x1024 or 4096x2048)
- **For best quality:** Use 8K texture (8192x4096) - larger file size
- **For faster loading:** Use 2K texture (2048x1024) - smaller file size
