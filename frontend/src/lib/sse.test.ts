import { describe, expect, it } from 'vitest';

import { SseParser } from './sse';

describe('SseParser', () => {
  it('parses the backend frame shape', () => {
    const parser = new SseParser();
    const out = parser.push('id: 4\nevent: next_question\ndata: {"seq":4,"state":"ASKING"}\n\n');
    expect(out).toEqual([{ id: '4', event: 'next_question', data: '{"seq":4,"state":"ASKING"}' }]);
  });

  it('ignores keepalive comments and blank dispatches', () => {
    const parser = new SseParser();
    expect(parser.push(': keepalive\n\n: keepalive\n\n')).toEqual([]);
  });

  it('reassembles messages split anywhere across chunks', () => {
    const frame = 'id: 7\nevent: grade_ready\ndata: {"seq":7}\n\n';
    for (let cut = 1; cut < frame.length; cut++) {
      const parser = new SseParser();
      const out = [...parser.push(frame.slice(0, cut)), ...parser.push(frame.slice(cut))];
      expect(out).toHaveLength(1);
      expect(out[0]).toEqual({ id: '7', event: 'grade_ready', data: '{"seq":7}' });
    }
  });

  it('handles CRLF and CR line endings, including a CRLF split across chunks', () => {
    const parser = new SseParser();
    const out = [
      ...parser.push('id: 1\r\nevent: a\r'),
      ...parser.push('\ndata: x\r\n\r\n'),
      ...parser.push('event: b\rdata: y\r\r'),
      // A trailing CR may be half of a CRLF: the next chunk settles it.
      ...parser.push(': keepalive\n'),
    ];
    expect(out).toEqual([
      { id: '1', event: 'a', data: 'x' },
      { id: null, event: 'b', data: 'y' },
    ]);
  });

  it('joins multi-line data and defaults the event name', () => {
    const parser = new SseParser();
    expect(parser.push('data: one\ndata: two\n\n')).toEqual([
      { id: null, event: 'message', data: 'one\ntwo' },
    ]);
  });

  it('does not carry an id over to the next message', () => {
    const parser = new SseParser();
    const out = parser.push('id: 3\nevent: x\ndata: 1\n\nevent: error\ndata: {}\n\n');
    expect(out.map((m) => m.id)).toEqual(['3', null]);
  });
});
