async function init() {
  const canvas = document.getElementById("gfx") as HTMLCanvasElement;
  if (!canvas) throw new Error("Canvas element not found");

  if (!navigator.gpu) throw new Error("WebGPU not supported");

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter found");

  const device = await adapter.requestDevice();
  const maxTextureSize = device.limits.maxTextureDimension2D;
  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();

  // ---------- SPHERE ----------
  function createSphere(radius: number, latBands: number, lonBands: number) {
    const positions: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];

    for (let lat = 0; lat <= latBands; lat++) {
      const theta = (lat * Math.PI) / latBands;
      const sinTheta = Math.sin(theta);
      const cosTheta = Math.cos(theta);

      for (let lon = 0; lon <= lonBands; lon++) {
        const phi = (lon * 2 * Math.PI) / lonBands;
        const sinPhi = Math.sin(phi);
        const cosPhi = Math.cos(phi);

        const x = cosPhi * sinTheta;
        const y = cosTheta;
        const z = sinPhi * sinTheta;
        positions.push(radius * x, radius * y, radius * z);
        
        // UV coordinates: u goes from 0 to 1 (longitude), v goes from 1 to 0 (latitude, flipped)
        uvs.push(1 - lon / lonBands, lat / latBands);
      }
    }

    for (let lat = 0; lat < latBands; lat++) {
      for (let lon = 0; lon < lonBands; lon++) {
        const first = lat * (lonBands + 1) + lon;
        const second = first + lonBands + 1;
        indices.push(first, first + 1, second);
        indices.push(second, first + 1, second + 1);
      }
    }

    return {
      positions: new Float32Array(positions),
      uvs: new Float32Array(uvs),
      indices: new Uint32Array(indices),
    };
  }

  const { positions, uvs, indices } = createSphere(1.0, 30, 30);
  console.log(`Sphere created: ${positions.length / 3} vertices, ${indices.length} indices`);

  const positionBuffer = device.createBuffer({
    size: positions.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(positionBuffer, 0, positions);

  const uvBuffer = device.createBuffer({
    size: uvs.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uvBuffer, 0, uvs);

  const indexBuffer = device.createBuffer({
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuffer, 0, indices);

  // ---------- UNIFORMS ----------
  // Uniform buffer: MVP matrix (64 bytes) + Light direction (16 bytes) = 80 bytes
  const uniformBuffer = device.createBuffer({
    size: 4 * 16 + 4 * 4, // MVP matrix + light direction (vec3 padded to vec4)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {},
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {},
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  // ---------- TEXTURE LOADING ----------
  async function loadTexture(url: string): Promise<GPUTexture> {
    const response = await fetch(url);
    const blob = await response.blob();
    const imageBitmap = await createImageBitmap(blob);

    const texture = device.createTexture({
      size: [imageBitmap.width, imageBitmap.height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    device.queue.copyExternalImageToTexture(
      { source: imageBitmap },
      { texture },
      [imageBitmap.width, imageBitmap.height]
    );

    return texture;
  }

  // Create a default texture (will be replaced when Earth texture loads)
  // For now, create a simple blue texture as fallback
  const defaultTexture = device.createTexture({
    size: [1, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: defaultTexture },
    new Uint8Array([100, 150, 255, 255]), // Blue color
    { bytesPerRow: 4 },
    [1, 1]
  );

  // Create sampler
  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "repeat",
    addressModeV: "clamp-to-edge",
  });

  // Load your downloaded texture
let earthTexture: GPUTexture;
try {
  earthTexture = await loadTexture("/earth_day.jpg"); // matches your downloaded file in public/
  console.log("Earth texture loaded from /earth_day.jpg");
} catch (err) {
  console.error("Failed to load earth_day.jpg, using default texture", err);
  earthTexture = defaultTexture;
}


  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: earthTexture.createView() },
      { binding: 2, resource: sampler },
    ],
  });

  // ---------- SHADERS ----------
  const shaderCode = `
  struct Uniforms {
    mvp : mat4x4<f32>,
    lightDir : vec3<f32>,
  };
  @group(0) @binding(0) var<uniform> uniforms : Uniforms;
  @group(0) @binding(1) var earthTexture : texture_2d<f32>;
  @group(0) @binding(2) var textureSampler : sampler;

  struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) worldPos : vec3<f32>,
    @location(1) uv : vec2<f32>,
  }

  @vertex
  fn vs_main(
    @location(0) position : vec3<f32>,
    @location(1) uv : vec2<f32>
  ) -> VertexOutput {
    var output : VertexOutput;
    output.worldPos = position;
    output.uv = uv;
    output.position = uniforms.mvp * vec4<f32>(position, 1.0);
    return output;
  }

  @fragment
  fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    // Sample the Earth texture
    let texColor = textureSample(earthTexture, textureSampler, input.uv);
    
    // For a sphere, the normal is just the normalized position
    let normal = normalize(input.worldPos);
    
    // Normalize light direction
    let lightDir = normalize(uniforms.lightDir);
    
    // Calculate diffuse lighting (Lambertian)
    let diffuse = max(dot(normal, lightDir), 0.0);
    
    // Add ambient lighting so dark side isn't completely black
    let ambient = 0.3;
    
    // Combine ambient and diffuse
    let lighting = ambient + (1.0 - ambient) * diffuse;
    
    // Apply lighting to texture color
    return vec4<f32>(texColor.rgb * lighting, texColor.a);
  }
  `;

  const shaderModule = device.createShaderModule({ code: shaderCode });

  // ---------- PIPELINE ----------
  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 3 * 4,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        },
        {
          arrayStride: 2 * 4,
          attributes: [{ shaderLocation: 1, offset: 0, format: "float32x2" }],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus",
    },
  });

  // ---------- RESIZE ----------
  let depthTexture: GPUTexture;
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.min(
      Math.floor(canvas.clientWidth * dpr),
      maxTextureSize,
    );
    const height = Math.min(
      Math.floor(canvas.clientHeight * dpr),
      maxTextureSize,
    );
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      context.configure({ device, format, alphaMode: "opaque" });
      depthTexture = device.createTexture({
        size: [width, height],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
  }
  window.addEventListener("resize", resize);
  resize();

  // ---------- CAMERA STATE ----------
  let cameraDistance = 3.0;
  let rotationX = 0; // Pitch (vertical rotation)
  let rotationY = 0; // Yaw (horizontal rotation)
  
  let isDragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;
  
  // Touch support
  let lastTouchDistance = 0;
  let isPinching = false;
  
  // Auto-rotation
  let autoRotateSpeed = 0.002; // radians per frame
  let isUserInteracting = false;
  let lastInteractionTime = 0;
  const autoRotateDelay = 2000; // ms before auto-rotate resumes after interaction

  // ---------- MATRIX FUNCTIONS ----------
  function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        out[i + j * 4] = 
          a[i + 0 * 4] * b[0 + j * 4] +
          a[i + 1 * 4] * b[1 + j * 4] +
          a[i + 2 * 4] * b[2 + j * 4] +
          a[i + 3 * 4] * b[3 + j * 4];
      }
    }
    return out;
  }

  // Create rotation matrix around X axis (pitch)
  function rotationXMatrix(angle: number): Float32Array {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([
      1, 0, 0, 0,
      0, c, s, 0,
      0, -s, c, 0,
      0, 0, 0, 1,
    ]);
  }

  // Create rotation matrix around Y axis (yaw)
  function rotationYMatrix(angle: number): Float32Array {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([
      c, 0, -s, 0,
      0, 1, 0, 0,
      s, 0, c, 0,
      0, 0, 0, 1,
    ]);
  }

  // Create translation matrix
  function translationMatrix(x: number, y: number, z: number): Float32Array {
    return new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      x, y, z, 1,
    ]);
  }

  function getMVPMatrix() {
    if (canvas.width === 0 || canvas.height === 0) {
      return new Float32Array([
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
      ]);
    }

    const aspect = canvas.width / canvas.height;
    const fov = Math.PI / 4;
    const near = 0.1;
    const far = 100;
    const f = 1 / Math.tan(fov / 2);

    // Perspective projection matrix (column-major)
    const proj = new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) / (near - far), -1,
      0, 0, (2 * far * near) / (near - far), 0,
    ]);

    // View matrix: rotate then translate
    // First rotate around Y (yaw), then X (pitch), then translate back by camera distance
    const rotY = rotationYMatrix(rotationY);
    const rotX = rotationXMatrix(rotationX);
    const translate = translationMatrix(0, 0, -cameraDistance);
    
    // Combine rotations: rotX * rotY (apply Y first, then X)
    const rotation = multiplyMat4(rotX, rotY);
    const view = multiplyMat4(translate, rotation);

    // MVP = proj * view
    return multiplyMat4(proj, view);
  }

  // ---------- MOUSE CONTROLS ----------
  canvas.addEventListener("mousedown", (e) => {
    isDragging = true;
    isUserInteracting = true;
    lastInteractionTime = Date.now();
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
    canvas.style.cursor = "grabbing";
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!isDragging) return;

    isUserInteracting = true;
    lastInteractionTime = Date.now();

    const deltaX = e.clientX - lastMouseX;
    const deltaY = e.clientY - lastMouseY;

    // Update rotation (sensitivity factor)
    rotationY += deltaX * 0.01;
    rotationX += deltaY * 0.01;

    // Clamp pitch to prevent flipping
    rotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationX));

    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });

  canvas.addEventListener("mouseup", () => {
    isDragging = false;
    canvas.style.cursor = "grab";
  });

  canvas.addEventListener("mouseleave", () => {
    isDragging = false;
    canvas.style.cursor = "grab";
  });

  // Zoom with mouse wheel (increased speed)
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    isUserInteracting = true;
    lastInteractionTime = Date.now();
    const zoomSpeed = 0.3; // Increased from 0.1
    cameraDistance += e.deltaY * zoomSpeed * 0.01;
    // Clamp zoom distance
    cameraDistance = Math.max(1.0, Math.min(10.0, cameraDistance));
  });

  // ---------- TOUCH CONTROLS ----------
  function getTouchDistance(touches: TouchList): number {
    if (touches.length < 2) return 0;
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  canvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    isUserInteracting = true;
    lastInteractionTime = Date.now();
    
    if (e.touches.length === 1) {
      isDragging = true;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      isPinching = true;
      lastTouchDistance = getTouchDistance(e.touches);
    }
  });

  canvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    isUserInteracting = true;
    lastInteractionTime = Date.now();
    
    if (e.touches.length === 1 && isDragging) {
      const deltaX = e.touches[0].clientX - lastMouseX;
      const deltaY = e.touches[0].clientY - lastMouseY;

      rotationY += deltaX * 0.01;
      rotationX += deltaY * 0.01;
      rotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationX));

      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    } else if (e.touches.length === 2 && isPinching) {
      const currentDistance = getTouchDistance(e.touches);
      const delta = lastTouchDistance - currentDistance;
      cameraDistance += delta * 0.01;
      cameraDistance = Math.max(1.0, Math.min(10.0, cameraDistance));
      lastTouchDistance = currentDistance;
    }
  });

  canvas.addEventListener("touchend", (e) => {
    e.preventDefault();
    if (e.touches.length === 0) {
      isDragging = false;
      isPinching = false;
    } else if (e.touches.length === 1) {
      isPinching = false;
      isDragging = true;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    }
  });

  // Set initial cursor style
  canvas.style.cursor = "grab";

  // ---------- RENDER LOOP ----------
  let frameCount = 0;
  function frame() {
    if (!depthTexture) {
      requestAnimationFrame(frame);
      return;
    }

    // Auto-rotate when user is not interacting
    const timeSinceInteraction = Date.now() - lastInteractionTime;
    if (timeSinceInteraction > autoRotateDelay && !isDragging && !isPinching) {
      rotationY += autoRotateSpeed;
    }

    const mvpMatrix = getMVPMatrix();
    
    // Light direction (pointing from top-right-front, will be normalized in shader)
    // This creates nice lighting that shows the 3D shape
    const lightDir = new Float32Array([0.5, 1.0, 0.5]);
    
    // Write MVP matrix (64 bytes) and light direction (16 bytes, vec3 padded to vec4)
    const uniformData = new Float32Array(16 + 4);
    uniformData.set(mvpMatrix, 0);
    uniformData.set(lightDir, 16);
    uniformData[19] = 0; // padding for vec4 alignment
    
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    
    // Debug: log first frame
    if (frameCount === 0) {
      console.log('First frame render');
      console.log('Canvas size:', canvas.width, 'x', canvas.height);
      console.log('Sphere vertices:', positions.length / 3);
      console.log('Sphere indices:', indices.length);
      console.log('First few positions:', Array.from(positions.slice(0, 9)));
      console.log('MVP matrix (first column):', Array.from(mvpMatrix.slice(0, 4)));
      console.log('MVP matrix (full):', Array.from(mvpMatrix));
    }
    frameCount++;

    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.setVertexBuffer(0, positionBuffer);
    renderPass.setVertexBuffer(1, uvBuffer);
    renderPass.setIndexBuffer(indexBuffer, "uint32");
    
    // Debug: verify we're actually drawing
    if (frameCount === 1) {
      console.log('Drawing', indices.length, 'indices');
    }
    
    renderPass.drawIndexed(indices.length);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

init();