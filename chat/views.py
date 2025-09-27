from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Conversation, Message
from django.db import transaction
from django.db.models import Count, Max
from django.utils.text import Truncator
import time, logging, json

import json
import httpx

log = logging.getLogger("synin")

#validator helper and use consistend timeouts
def _validate_messages(msgs):
    if not isinstance(msgs, list) or not msgs:
        return "messages must be a non-empty list"
    if len(msgs) > settings.HISTORY_MAX_TURNS:
        return f"too many turns (>{settings.HISTORY_MAX_TURNS})"
    for m in msgs:
        role = m.get("role")
        content = m.get("content", "")
        if role not in {"system", "user", "assistant"}:
            return "invalid role"
        if not isinstance(content, str):
            return "content must be string"
        if len(content) > settings.MESSAGE_MAX_CHARS:
            return f"message too long (>{settings.MESSAGE_MAX_CHARS} chars)"
    return None
#conversation 
def _get_or_create_conversation(conv_id: str | None, session_key: str | None) -> Conversation:
    if conv_id:
        try:
            return Conversation.objects.get(id=conv_id)
        except Conversation.DoesNotExist:
            pass
    return Conversation.objects.create(session_key=session_key or "")

def _store_incoming_messages(conv: Conversation, messages_in):
    existing = conv.messages.filter(role__in={"system", "user"}).count()
    for m in messages_in:
        if m.get("role") in {"system", "user"}:
            if existing > 0:
                existing -= 1
                continue
            Message.objects.create(conversation=conv, role=m["role"], content=m["content"])

def _maybe_set_title(conv: Conversation):
    # set a simple title from first user message if empty
    if not conv.title:
        first = conv.messages.filter(role="user").first()
        if first:
            conv.title = (first.content[:60] + "…") if len(first.content) > 60 else first.content
            conv.save(update_fields=["title"])

@api_view(["GET", "DELETE"])
def get_conversation(request, conv_id):
    if not request.session.session_key:
        request.session.save()
    session_key = request.session.session_key or ""
    try:
        conv = Conversation.objects.get(id=conv_id)
    except Conversation.DoesNotExist:
        return Response({"error": "not found"}, status=404)
    if conv.session_key and conv.session_key != session_key:
        return Response({"error": "not found"}, status=404)
    if not conv.session_key and session_key:
        conv.session_key = session_key
        conv.save(update_fields=["session_key"])
    if request.method == 'DELETE':
        conv.delete()
        return Response({"ok": True})
    msgs = [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in conv.messages.all()]
    return Response({"id": str(conv.id), "title": conv.title, "messages": msgs})

@api_view(["POST"])
def clear_conversation(request, conv_id):
    if not request.session.session_key:
        request.session.save()
    session_key = request.session.session_key or ""
    try:
        conv = Conversation.objects.get(id=conv_id)
    except Conversation.DoesNotExist:
        return Response({"error": "not found"}, status=404)
    if conv.session_key and conv.session_key != session_key:
        return Response({"error": "not found"}, status=404)
    if not conv.session_key and session_key:
        conv.session_key = session_key
        conv.save(update_fields=["session_key"])
    conv.messages.all().delete()
    conv.title = ""
    conv.save(update_fields=["title"])
    return Response({"ok": True})


@api_view(["GET"])
def list_conversations(request):
    if not request.session.session_key:
        request.session.save()
    session_key = request.session.session_key or ""
    qs = (Conversation.objects
          .filter(session_key=session_key)
          .annotate(
              message_count=Count("messages"),
              last_message_at=Max("messages__created_at"),
          )
          .order_by("-last_message_at", "-created_at"))
    data = []
    for conv in qs:
        data.append({
            "id": str(conv.id),
            "title": conv.title or "",
            "created_at": conv.created_at.isoformat(),
            "last_message_at": conv.last_message_at.isoformat() if conv.last_message_at else None,
            "message_count": conv.message_count,
        })
    return Response({"data": data})

@api_view(["POST"])
def optimize_prompt(request):
    if not request.session.session_key:
        request.session.save()
    prompt_text = (request.data.get("prompt") or "").strip()
    if not prompt_text:
        return Response({"error": "prompt is required"}, status=400)

    model = request.data.get("model") or settings.DEFAULT_MODEL_ID
    if not model:
        return Response({"error": "model is required"}, status=400)

    try:
        temperature = float(request.data.get("temperature", 0.2))
    except (TypeError, ValueError):
        temperature = 0.2
    try:
        top_p = float(request.data.get("top_p", 1.0))
    except (TypeError, ValueError):
        top_p = 1.0

    max_tokens = request.data.get("max_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            return Response({"error": "max_tokens must be integer"}, status=400)
        if max_tokens > settings.MAX_TOKENS_LIMIT:
            max_tokens = settings.MAX_TOKENS_LIMIT

    instructions = (
        "You are an expert prompt engineer. Rewrite the provided prompt so the model response is clear, "
        "well-scoped, and actionable. Return strict JSON with keys `optimized_prompt` (string) and `notes` "
        "(array of short strings describing key improvements). Do not include any other text."
    )

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt_text},
    ]

    payload = {"model": model, "messages": messages, "temperature": temperature, "top_p": top_p}
    if max_tokens:
        payload["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}
    if settings.LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LM_STUDIO_API_KEY}"

    url = f"{settings.LM_STUDIO_BASE_URL}/chat/completions"
    timeout = httpx.Timeout(
        connect=settings.CONNECT_TIMEOUT,
        read=settings.REQUEST_TIMEOUT,
        write=settings.REQUEST_TIMEOUT,
        pool=settings.CONNECT_TIMEOUT,
    )

    try:
        t0 = time.time()
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        lm = resp.json()
        dt_ms = (time.time() - t0) * 1000.0
    except httpx.HTTPError as exc:
        log.warning("LMStudio optimize error: %s", exc)
        return Response({"error": str(exc)}, status=status.HTTP_502_BAD_GATEWAY)

    content = (
        lm.get("choices", [{}])[0].get("message", {}).get("content")
        if isinstance(lm, dict) else ""
    ) or ""

    optimized_prompt = prompt_text
    notes = []

    if content:
        raw = content.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            optimized_prompt = raw or prompt_text
        else:
            optimized_prompt = (parsed.get("optimized_prompt") or parsed.get("optimizedPrompt") or raw).strip() or prompt_text
            parsed_notes = parsed.get("notes") or parsed.get("tips") or []
            if isinstance(parsed_notes, str):
                notes = [parsed_notes.strip()] if parsed_notes.strip() else []
            elif isinstance(parsed_notes, list):
                notes = [str(item).strip() for item in parsed_notes if str(item).strip()]

    log.info("Prompt optimize %.0fms | prompt=%s", dt_ms, Truncator(prompt_text).chars(120))

    return Response({
        "ok": True,
        "model": model,
        "optimized_prompt": optimized_prompt.strip(),
        "notes": notes,
    })



# ---------- Web page ----------
def index(request):
    # Looks for templates/index.html because settings.TEMPLATES includes BASE_DIR / 'templates'
    return render(request, "index.html")


# ---------- Simple health check ----------
def health(request):
    return JsonResponse({"status": "ok", "app": "synin", "version": 1})


# ---------- API: list models (proxy to LM Studio) ----------
@csrf_exempt
@api_view(["GET"])
def list_models(request):
    headers = {"Content-Type": "application/json"}
    if settings.LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LM_STUDIO_API_KEY}"

    url = f"{settings.LM_STUDIO_BASE_URL}/models"
    try:
        with httpx.Client(timeout=60) as client:
            r = client.get(url, headers=headers)
        r.raise_for_status()
        return Response(r.json())
    except httpx.HTTPError as e:
        return Response({"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)


# ---------- API: non-streaming chat ----------
@csrf_exempt
@api_view(["POST"])
def chat_completion(request):
    data = request.data or {}
    messages_in   = data.get("messages", [])
    model         = data.get("model") or settings.DEFAULT_MODEL_ID
    temperature   = float(data.get("temperature", 0.2))
    top_p         = data.get("top_p", 1.0)
    max_tokens    = data.get("max_tokens")
    conversation_id = data.get("conversation_id")
    if not request.session.session_key:
        request.session.save()
    session_key   = request.session.session_key or ""

    # Validate
    err = _validate_messages(messages_in)
    if err:
        return Response({"error": err}, status=400)
    if not model:
        return Response({"error": "model is required"}, status=400)

    # Cap max_tokens
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except Exception:
            return Response({"error": "max_tokens must be integer"}, status=400)
        if max_tokens > settings.MAX_TOKENS_LIMIT:
            max_tokens = settings.MAX_TOKENS_LIMIT

    # Persist incoming user/system turns
    with transaction.atomic():
        conv = _get_or_create_conversation(conversation_id, session_key)
        _store_incoming_messages(conv, messages_in)

    headers = {"Content-Type": "application/json"}
    if settings.LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LM_STUDIO_API_KEY}"

    payload = {"model": model, "messages": messages_in, "temperature": temperature, "top_p": top_p}
    if max_tokens:
        payload["max_tokens"] = max_tokens

    url = f"{settings.LM_STUDIO_BASE_URL}/chat/completions"

    # httpx timeout object
    timeout = httpx.Timeout(
        connect=settings.CONNECT_TIMEOUT,
        read=settings.REQUEST_TIMEOUT,
        write=settings.REQUEST_TIMEOUT,
        pool=settings.CONNECT_TIMEOUT,
    )

    try:
        t0 = time.time()
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        lm = r.json()
        dt_ms = (time.time() - t0) * 1000.0

        content = lm.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

        # Save assistant
        with transaction.atomic():
            Message.objects.create(conversation=conv, role="assistant", content=content.strip())
            _maybe_set_title(conv)

        log.info("LMStudio chat %.0fms | prompt=%s",
                 dt_ms,
                 Truncator(messages_in[-1]["content"]).chars(120))

        return Response({
            "ok": True,
            "model": model,
            "assistant": content.strip(),
            "conversation_id": str(conv.id),
        })

    except httpx.HTTPError as e:
        log.warning("LMStudio error: %s", e)
        return Response({"ok": False, "error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)


# ---------- API: streaming chat (SSE-style) ----------
@csrf_exempt
def chat_stream(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    messages_in     = data.get("messages", [])
    model           = data.get("model") or settings.DEFAULT_MODEL_ID
    temperature     = float(data.get("temperature", 0.2))
    top_p           = data.get("top_p", 1.0)
    max_tokens      = data.get("max_tokens")
    conversation_id = data.get("conversation_id")
    if not request.session.session_key:
        request.session.save()
    session_key     = request.session.session_key or ""

    # Validate
    err = _validate_messages(messages_in)
    if err:
        return JsonResponse({"error": err}, status=400)
    if not model:
        return JsonResponse({"error": "model is required"}, status=400)

    # Cap max_tokens
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except Exception:
            return JsonResponse({"error": "max_tokens must be integer"}, status=400)
        if max_tokens > settings.MAX_TOKENS_LIMIT:
            max_tokens = settings.MAX_TOKENS_LIMIT

    # Persist incoming user/system turns first
    with transaction.atomic():
        conv = _get_or_create_conversation(conversation_id, session_key)
        _store_incoming_messages(conv, messages_in)

    headers = {"Content-Type": "application/json"}
    if settings.LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LM_STUDIO_API_KEY}"

    payload = {
        "model": model,
        "messages": messages_in,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    url = f"{settings.LM_STUDIO_BASE_URL}/chat/completions"

    def event_stream():
        reply_accum = []
        log.info("LMStudio stream start | prompt=%s",
                 Truncator(messages_in[-1]["content"]).chars(120))
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream("POST", url, json=payload, headers=headers) as r:
                    r.raise_for_status()
                    for raw in r.iter_lines():
                        if not raw:
                            continue
                        try:
                            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                        except Exception:
                            text = str(raw)

                        line = text.strip()
                        if line.startswith("data:"):
                            line = line[5:].strip()

                        # collect for DB save
                        try:
                            j = json.loads(line)
                            delta = j.get("choices", [{}])[0].get("delta", {}).get("content")
                            if delta:
                                reply_accum.append(delta)
                        except Exception:
                            pass

                        # forward to client with exactly one "data:"
                        yield f"data: {line}\n\n"
        finally:
            full = "".join(reply_accum).strip()
            if full:
                with transaction.atomic():
                    Message.objects.create(conversation=conv, role="assistant", content=full)
                    _maybe_set_title(conv)
                log.info("LMStudio stream end | %d chars", len(full))
            yield "data: [DONE]\n\n"

    resp = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    resp["X-Conversation-Id"] = str(conv.id)
    return resp


