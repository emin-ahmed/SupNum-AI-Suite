import React, { useState } from 'react';
import { Plus, MessageSquare, Settings, User, Menu, X, Trash2, Edit3 } from 'lucide-react';

export const Sidebar = ({
  chats,
  currentChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onRenameChat
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState('');

  const handleRename = (chatId, currentTitle) => {
    setEditingId(chatId);
    setEditTitle(currentTitle);
  };

  const saveRename = () => {
    if (editingId && editTitle.trim()) {
      onRenameChat(editingId, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle('');
  };

  const cancelRename = () => {
    setEditingId(null);
    setEditTitle('');
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-gray-800 text-white rounded-md hover:bg-gray-700 transition-colors"
      >
        {isOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}

      <div 
      style={{ backgroundColor: '#0077B6' }}
      className={`
        fixed lg:static inset-y-0 left-0 z-40 w-64 text-white flex flex-col
        transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        {/* Header */}
        <div className="p-4 border-b border-gray-700 space-y-2">

          {/* FAQ Button with Tooltip */}
          <button
            onClick={() => {
              onNewChat('faq');
              setIsOpen(false);
            }}
            // style={{ backgroundColor: '#0096C7' }}
            title="Posez vos questions administratives (inscription, calendrier...)"
            className="w-full flex items-center bg-[#0096C7] gap-3 px-4 py-3  hover:bg-[#023E8A] rounded-lg transition-colors"
          >
            <Plus size={18} />
            <span className="font-medium">FAQ</span>
          </button>

          {/* Advisor Button with Tooltip */}
          <button
            onClick={() => {
              onNewChat('advisor');
              setIsOpen(false);
            }}
            // style={{ backgroundColor: '#0096C7' }}
            title="Recevez des conseils sur vos études, votre orientation ou carrière"
            className="w-full flex items-center gap-3 bg-[#0096C7] px-4 py-3  hover:bg-[#023E8A] rounded-lg transition-colors"
          >
            <Plus size={18} />
            <span className="font-medium">Advisor</span>
          </button>
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-2">
          <div className="space-y-1">
            {chats.map((chat) => (
              <div
                key={chat.id}
                className={`group relative flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                  currentChatId === chat.id ? 'bg-gray-700' : 'hover:bg-gray-800'
                }`}
                onClick={() => {
                  onSelectChat(chat.id);
                  setIsOpen(false);
                }}
              >
                <MessageSquare size={16} className="flex-shrink-0" />

                {editingId === chat.id ? (
                  <input
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onBlur={saveRename}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') saveRename();
                      if (e.key === 'Escape') cancelRename();
                    }}
                    className="flex-1 bg-gray-600 text-white px-2 py-1 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                    autoFocus
                  />
                ) : (
                  <span className="flex-1 text-sm truncate">{chat.title}</span>
                )}

                {editingId !== chat.id && (
                  <div className="opacity-0 group-hover:opacity-100 flex gap-1 transition-opacity">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRename(chat.id, chat.title);
                      }}
                      className="p-1 hover:bg-gray-600 rounded"
                    >
                      <Edit3 size={12} />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteChat(chat.id);
                      }}
                      className="p-1 hover:bg-gray-600 rounded text-red-400"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        {/* <div className="p-4 border-t border-gray-700">
          <div className="flex items-center gap-3 px-3 py-2 hover:bg-gray-800 rounded-lg cursor-pointer transition-colors">
            <User size={16} />
            <span className="text-sm">Mon compte</span>
          </div>
          <div className="flex items-center gap-3 px-3 py-2 hover:bg-gray-800 rounded-lg cursor-pointer transition-colors">
            <Settings size={16} />
            <span className="text-sm">Paramètres</span>
          </div>
        </div> */}
      </div>
    </>
  );
};
