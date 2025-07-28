import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip } from 'lucide-react';

export const ChatInput = ({ onSendMessage, disabled = false }) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  return (
    <div className="border-t border-gray-200 bg-white">
      <div className="max-w-4xl mx-auto px-4 py-4">
        <form onSubmit={handleSubmit} className="relative">
          <div className="flex items-end gap-3 bg-white border border-gray-300 rounded-xl shadow-sm focus-within:border-gray-400 focus-within:shadow-md transition-all">
            <button
              type="button"
              className="flex-shrink-0 p-3 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <Paperclip size={20} />
            </button>
            
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Envoyez un message à ChatGPT..."
              disabled={disabled}
              rows={1}
              className="flex-1 resize-none border-none outline-none py-3 text-gray-900 placeholder-gray-500 disabled:opacity-50 disabled:cursor-not-allowed max-h-32 overflow-y-auto"
              style={{ minHeight: '24px' }}
            />
            
            <button
              type="submit"
              disabled={!input.trim() || disabled}
              className="flex-shrink-0 p-3 text-gray-400 hover:text-gray-600 disabled:text-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              <Send size={20} />
            </button>
          </div>
          
          <p className="text-xs text-gray-500 text-center mt-2">
            ChatGPT peut faire des erreurs. Vérifiez les informations importantes.
          </p>
        </form>
      </div>
    </div>
  );
};