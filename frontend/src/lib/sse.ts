/**
 * A minimal Server-Sent Events parser (the WHATWG "event stream" format), for
 * streams read with `fetch` - `EventSource` cannot send the bearer token.
 *
 * The backend's frames (backend/app/interview/stream.py) are:
 *
 *     id: <seq>
 *     event: <name>
 *     data: <one line of JSON>
 *     <blank line>
 *
 * plus `: keepalive` comment lines. This parser handles the general format
 * anyway (CRLF/CR line ends, multi-line data, chunks split anywhere), so a
 * proxy re-chunking the stream cannot corrupt an event.
 */

export interface SseMessage {
  /** The `id:` field, or null when the message had none (e.g. `event: error`). */
  id: string | null;
  /** The `event:` field; "message" when absent, per the spec. */
  event: string;
  data: string;
}

export class SseParser {
  private buffer = '';
  private id: string | null = null;
  private event = '';
  private data: string[] = [];

  /** Feed one decoded chunk; returns every message it completed. */
  push(chunk: string): SseMessage[] {
    this.buffer += chunk;
    const out: SseMessage[] = [];
    // A trailing "\r" might be the first half of "\r\n": keep it for the next chunk.
    let match: RegExpExecArray | null;
    const lineEnd = /\r\n|\n|\r(?!$)/g;
    let start = 0;
    while ((match = lineEnd.exec(this.buffer)) !== null) {
      const line = this.buffer.slice(start, match.index);
      start = match.index + match[0].length;
      const message = this.line(line);
      if (message) out.push(message);
    }
    this.buffer = this.buffer.slice(start);
    return out;
  }

  private line(line: string): SseMessage | null {
    if (line === '') {
      // Blank line: dispatch, if anything was collected.
      if (this.data.length === 0 && !this.event) {
        this.event = '';
        return null;
      }
      const message: SseMessage = {
        id: this.id,
        event: this.event || 'message',
        data: this.data.join('\n'),
      };
      this.id = null;
      this.event = '';
      this.data = [];
      return message;
    }
    if (line.startsWith(':')) return null; // a comment (keepalive)
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'data') this.data.push(value);
    else if (field === 'event') this.event = value;
    else if (field === 'id' && !value.includes('\0')) this.id = value;
    // "retry" and unknown fields are ignored.
    return null;
  }
}
