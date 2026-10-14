/**
 * The question's wording, exactly as the server sent it. Line breaks and
 * indentation are preserved; a ``` fenced block is shown as code. Rendered as
 * text, never as HTML.
 */

import { splitFences } from './content';

export function QuestionText({ text }: { text: string }) {
  return (
    <div className="space-y-3 text-base leading-relaxed">
      {splitFences(text).map((segment, index) =>
        segment.code ? (
          <pre
            key={index}
            className="overflow-x-auto rounded-md border bg-muted px-3 py-2 font-mono text-sm leading-normal"
          >
            <code>{segment.text}</code>
          </pre>
        ) : (
          <p key={index} className="whitespace-pre-wrap break-words">
            {segment.text.replace(/^\n+|\n+$/g, '')}
          </p>
        ),
      )}
    </div>
  );
}
