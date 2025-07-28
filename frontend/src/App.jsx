import React, { useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import { TypingIndicator } from './components/TypingIndicator';
import { ChatInput } from './components/ChatInput';
import { WelcomeScreen } from './components/WelcomeScreen';
import { useChatGPT } from './hooks/useChatGPT';

function App() {
  const {
    chats,
    currentChat,
    currentChatId,
    isTyping,
    createNewChat,
    selectChat,
    deleteChat,
    renameChat,
    sendMessage,
  } = useChatGPT();

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentChat?.messages, isTyping]);

  const handleSendMessage = (message) => {
    if (!currentChatId) {
      createNewChat();
      // Wait for the new chat to be created, then send the message
      setTimeout(() => sendMessage(message), 100);
    } else {
      sendMessage(message);
    }
  };

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <Sidebar
        chats={chats}
        currentChatId={currentChatId}
        onNewChat={createNewChat}
        onSelectChat={selectChat}
        onDeleteChat={deleteChat}
        onRenameChat={renameChat}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col lg:ml-0">
        {!currentChat ? (
          <WelcomeScreen onSendMessage={handleSendMessage} />
        ) : (
          <>
            {/* Messages Container */}
            <div className="flex-1 overflow-y-auto">
              {currentChat.messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message.text}
                  isBot={message.isBot}
                  timestamp={message.timestamp}
                />
              ))}
              
              {isTyping && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>
          </>
        )}

        {/* Input Area */}
        <ChatInput onSendMessage={handleSendMessage} disabled={isTyping} />
      </div>
    </div>
  );
}

export default App;