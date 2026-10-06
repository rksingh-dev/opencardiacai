"use client";

import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export default function ThreeDViewer({
  points,
  colors
}: {
  points: Float32Array;
  colors: Float32Array;
}) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mountRef.current) return;
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 250;
    camera.position.y = -100;
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 2.0;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(points, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 1.5,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9
    });

    const pointCloud = new THREE.Points(geometry, material);
    
    // Center the cloud
    geometry.computeBoundingBox();
    if (geometry.boundingBox) {
      const center = new THREE.Vector3();
      geometry.boundingBox.getCenter(center);
      pointCloud.position.sub(center);
    }
    
    // Standard medical orientation (flip Y)
    pointCloud.rotation.x = Math.PI;

    scene.add(pointCloud);

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!mountRef.current) return;
      const w = mountRef.current.clientWidth;
      const h = mountRef.current.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      if (mountRef.current) {
        mountRef.current.innerHTML = '';
      }
    };
  }, [points, colors]);

  return (
    <div 
      ref={mountRef} 
      style={{ 
        width: '100%', 
        height: '400px', 
        background: 'radial-gradient(circle, #1a1a1a 0%, #000000 100%)', 
        borderRadius: '8px', 
        overflow: 'hidden',
        border: '1px solid rgba(255,255,255,0.1)'
      }} 
    />
  );
}
