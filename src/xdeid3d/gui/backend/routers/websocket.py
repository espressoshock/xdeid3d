import asyncio
import json
from typing import Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.spherehead_service import spherehead_service
from services.pti_service import pti_service
from services.stable_diffusion_service import stable_diffusion_service

router = APIRouter()

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}


@router.websocket("/generation/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    # Close any existing connection with same session_id
    if session_id in active_connections:
        try:
            old_ws = active_connections[session_id]
            await old_ws.close()
            print(f"[WebSocket] Closed old connection for {session_id}")
        except:
            pass
        del active_connections[session_id]

    await websocket.accept()
    active_connections[session_id] = websocket
    print(f"[WebSocket] Client connected: {session_id}")

    try:
        while True:
            # Wait for messages from client
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                message_type = message.get("type")

                print(f"[WebSocket] Received: {message_type} from {session_id}")

                if message_type == "ping":
                    # Respond to ping with pong
                    try:
                        await websocket.send_json({"type": "pong"})
                        print(f"[WebSocket] Sent pong to {session_id}")
                    except Exception as e:
                        print(f"[WebSocket] ERROR sending pong: {type(e).__name__}: {e}")
                        raise

                elif message_type == "generate_seed":
                    await handle_seed_generation(websocket, session_id, message)

                elif message_type == "generate_text":
                    await handle_text_generation(websocket, session_id, message)

                elif message_type == "start_pti":
                    await handle_pti_projection(websocket, session_id, message)

            except json.JSONDecodeError as e:
                print(f"[WebSocket] JSON decode error: {e}")
                continue

    except WebSocketDisconnect as e:
        print(f"[WebSocket] Client disconnected: {session_id} (code: {getattr(e, 'code', 'unknown')})")
    except Exception as e:
        print(f"[WebSocket] Error for {session_id}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if session_id in active_connections:
            del active_connections[session_id]
        print(f"[WebSocket] Cleaned up connection: {session_id}")


async def handle_seed_generation(websocket: WebSocket, session_id: str, message: dict):
    """Handle seed-based generation using real SphereHead service"""
    try:
        seed = message.get("seed", 0)
        params = message.get("params", {})

        print(f"[WebSocket] Starting seed generation: seed={seed}, params={params}")

        # Extract parameters with defaults
        truncation = params.get("truncation", 0.65)
        nrr = params.get("nrr", 128)
        sample_mult = params.get("sampleMult", 1.5)

        # Call real SphereHead service
        result = await spherehead_service.generate_from_seed(
            session_id=session_id,
            seed=seed,
            truncation=truncation,
            nrr=nrr,
            sample_mult=sample_mult,
            websocket=websocket,
        )

        print(f"[WebSocket] Generation completed for seed {seed}")

    except Exception as e:
        print(f"[WebSocket] Generation error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "error": str(e),
            "code": "GENERATION_ERROR"
        })


async def handle_text_generation(websocket: WebSocket, session_id: str, message: dict):
    """
    Handle text-to-3D generation pipeline: Text → Stable Diffusion → PTI → 3D Head

    This is a two-stage process:
    Stage 1: Generate face image from text prompt using Stable Diffusion (~30-60s)
    Stage 2: Convert generated image to 3D using PTI (~5-10 minutes)
    """
    try:
        # Extract Stable Diffusion parameters
        prompt = message.get("prompt", "")
        negative_prompt = message.get("negative_prompt", "")
        steps = message.get("steps", 50)
        guidance_scale = message.get("guidance_scale", 7.5)

        # Extract PTI parameters (from frontend or use defaults)
        pti_params = message.get("pti_params", {})
        w_steps = pti_params.get("w_steps", 500)
        pti_steps = pti_params.get("pti_steps", 350)
        truncation = pti_params.get("truncation", 0.65)
        nrr = pti_params.get("nrr", 128)
        sample_mult = pti_params.get("sampleMult", 1.5)
        generate_video = pti_params.get("generate_video", False)
        optimize_noise = pti_params.get("optimize_noise", False)
        initial_noise_factor = pti_params.get("initial_noise_factor", 0.05)
        noise_ramp_length = pti_params.get("noise_ramp_length", 0.75)
        regularize_noise_weight = pti_params.get("regularize_noise_weight", 1e5)

        print(f"[WebSocket] Text-to-3D generation starting for session {session_id}")
        print(f"[WebSocket] Prompt: {prompt}")
        print(f"[WebSocket] SD params: steps={steps}, guidance={guidance_scale}")
        print(f"[WebSocket] PTI params: w_steps={w_steps}, pti_steps={pti_steps}, nrr={nrr}")

        # ====== STAGE 1: Stable Diffusion Generation ======
        await websocket.send_json({
            "type": "progress",
            "stage": "sd_generation",
            "progress": 0.0,
            "step": 0,
            "total_steps": steps,
            "message": "Stage 1/2: Generating face image with Stable Diffusion...",
        })

        # Generate face image using Stable Diffusion
        sd_result = await stable_diffusion_service.generate_face(
            session_id=session_id,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            steps=steps,
            guidance_scale=guidance_scale,
            websocket=websocket,
        )

        print(f"[WebSocket] Stage 1 complete. Image saved to: {sd_result['image_path']}")

        # Send intermediate result with generated image
        await websocket.send_json({
            "type": "progress",
            "stage": "sd_generation",
            "progress": 1.0,
            "message": "Stage 1/2 complete! Starting 3D conversion...",
            "preview_image": f"/api/media/{session_id}/sd_generated.png",
        })

        # ====== STAGE 2: PTI Projection (Image → 3D) ======
        await websocket.send_json({
            "type": "progress",
            "stage": "w_projection",
            "progress": 0.0,
            "step": 0,
            "total_steps": w_steps + pti_steps,  # Total PTI steps
            "message": "Stage 2/2: Converting image to 3D with PTI...",
        })

        # Call PTI service with generated image and user-specified parameters
        pti_result = await pti_service.project_image(
            session_id=session_id,
            upload_id=sd_result["upload_id"],
            w_steps=w_steps,
            pti_steps=pti_steps,
            truncation=truncation,
            nrr=nrr,
            sample_mult=sample_mult,
            generate_video=generate_video,
            optimize_noise=optimize_noise,
            initial_noise_factor=initial_noise_factor,
            noise_ramp_length=noise_ramp_length,
            regularize_noise_weight=regularize_noise_weight,
            websocket=websocket,
        )

        print(f"[WebSocket] Text-to-3D pipeline complete for session {session_id}")

        # Note: PTI service already sends the final "complete" message
        # So we don't need to send another one here

    except Exception as e:
        print(f"[WebSocket] Text-to-3D generation error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "error": str(e),
            "code": "TEXT_GENERATION_ERROR"
        })


async def handle_pti_projection(websocket: WebSocket, session_id: str, message: dict):
    """Handle PTI projection using real pti_service"""
    try:
        upload_id = message.get("upload_id", "")
        params = message.get("params", {})

        print(f"[WebSocket] Starting PTI projection: upload_id={upload_id}, params={params}")

        # Extract parameters with defaults
        w_steps = params.get("w_steps", 500)
        pti_steps = params.get("pti_steps", 350)
        truncation = params.get("truncation", 0.7)
        nrr = params.get("nrr", 128)
        sample_mult = params.get("sampleMult", 1.5)
        generate_video = params.get("generate_video", True)

        # Noise optimization parameters
        optimize_noise = params.get("optimize_noise", False)
        initial_noise_factor = params.get("initial_noise_factor", 0.05)
        noise_ramp_length = params.get("noise_ramp_length", 0.75)
        regularize_noise_weight = params.get("regularize_noise_weight", 1e5)

        # Call real PTI service
        result = await pti_service.project_image(
            session_id=session_id,
            upload_id=upload_id,
            w_steps=w_steps,
            pti_steps=pti_steps,
            truncation=truncation,
            nrr=nrr,
            sample_mult=sample_mult,
            generate_video=generate_video,
            optimize_noise=optimize_noise,
            initial_noise_factor=initial_noise_factor,
            noise_ramp_length=noise_ramp_length,
            regularize_noise_weight=regularize_noise_weight,
            websocket=websocket,
        )

        print(f"[WebSocket] PTI projection completed for upload_id {upload_id}")

    except Exception as e:
        print(f"[WebSocket] PTI projection error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "error": str(e),
            "code": "PTI_ERROR"
        })


async def send_to_session(session_id: str, message: dict):
    """Send message to a specific session"""
    if session_id in active_connections:
        try:
            await active_connections[session_id].send_json(message)
        except Exception as e:
            print(f"[WebSocket] Failed to send to {session_id}: {e}")
