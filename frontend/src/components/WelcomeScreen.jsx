import React from 'react';
import { MessageSquare, Lightbulb, Code, BookOpen } from 'lucide-react';

export const WelcomeScreen = ({ onSendMessage }) => {
  const suggestions = [
    {
      icon: <Lightbulb size={20} />,
      title: "Expliquer des concepts",
      description: "Expliquez-moi la physique quantique de manière simple",
      prompt: "Expliquez-moi la physique quantique de manière simple"
    },
    {
      icon: <Code size={20} />,
      title: "Aide au codage",
      description: "Comment créer une API REST avec Node.js ?",
      prompt: "Comment créer une API REST avec Node.js ?"
    },
    {
      icon: <BookOpen size={20} />,
      title: "Rédaction créative",
      description: "Écrivez une histoire courte sur l'espace",
      prompt: "Écrivez une histoire courte sur l'espace"
    },
    {
      icon: <MessageSquare size={20} />,
      title: "Conversation générale",
      description: "Parlons de vos centres d'intérêt",
      prompt: "Parlons de vos centres d'intérêt"
    }
  ];

  return (
    <div className="flex-1 flex items-center justify-center p-4">
      <div className="max-w-4xl w-full">
        <div className="text-center mb-12">
          <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-6">
            <MessageSquare size={32} className="text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Comment puis-je vous aider aujourd'hui ?
          </h1>
          <p className="text-lg text-gray-600">
            Posez-moi n'importe quelle question ou choisissez une suggestion ci-dessous
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-3xl mx-auto">
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => onSendMessage(suggestion.prompt)}
              className="p-6 bg-white border border-gray-200 rounded-xl hover:border-gray-300 hover:shadow-md transition-all text-left group"
            >
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center text-gray-600 group-hover:bg-gray-200 transition-colors">
                  {suggestion.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-gray-900 mb-1">
                    {suggestion.title}
                  </h3>
                  <p className="text-sm text-gray-600">
                    {suggestion.description}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};