import React from 'react';
import { User, Bot, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';

export const ChatMessage = ({ message, isBot, timestamp }) => {
  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(message);
  };

  return (
    <div className={`group w-full ${isBot ? 'bg-gray-50' : 'bg-white'} border-b border-gray-100`}>
      <div className="max-w-4xl mx-auto px-4 py-6">
        <div className="flex gap-4">
          {/* Avatar */}
          <div className={`
            flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
            ${isBot ? 'bg-green-500' : 'bg-blue-500'}
          `}>
            {isBot ? (
              <Bot size={16} className="text-white" />
            ) : (
              <User size={16} className="text-white" />
            )}
          </div>

          {/* Message Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <span className="font-semibold text-gray-900">
                {isBot ? 'ChatGPT' : 'Vous'}
              </span>
              <span className="text-xs text-gray-500">
                {formatTime(timestamp)}
              </span>
            </div>

            <div className="prose prose-gray max-w-none">
              <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
                {message}
              </p>
            </div>

            {/* Action buttons for bot messages */}
            {isBot && (
              <div className="flex items-center gap-2 mt-4 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={copyToClipboard}
                  className="p-1.5 hover:bg-gray-200 rounded-md transition-colors"
                  title="Copier"
                >
                  <Copy size={14} className="text-gray-600" />
                </button>
                <button
                  className="p-1.5 hover:bg-gray-200 rounded-md transition-colors"
                  title="Bonne réponse"
                >
                  <ThumbsUp size={14} className="text-gray-600" />
                </button>
                <button
                  className="p-1.5 hover:bg-gray-200 rounded-md transition-colors"
                  title="Mauvaise réponse"
                >
                  <ThumbsDown size={14} className="text-gray-600" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};