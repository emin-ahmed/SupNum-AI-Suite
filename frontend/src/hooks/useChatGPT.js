import { useState, useCallback } from 'react';

const botResponses = [
  "Je comprends votre question. Laissez-moi vous expliquer cela en détail...",
  "C'est une excellente question ! Voici ce que je peux vous dire à ce sujet...",
  "Permettez-moi de vous donner une réponse complète et structurée...",
  "Je vais aborder votre demande sous plusieurs angles pour vous donner la meilleure réponse possible...",
  "Voici une explication détaillée qui devrait répondre à votre question...",
  "C'est un sujet fascinant ! Laissez-moi vous expliquer les points clés...",
  "Je vais vous donner une réponse précise et utile sur ce sujet...",
  "Excellente question ! Voici ce que vous devez savoir...",
  "Permettez-moi de vous fournir une réponse complète et bien structurée...",
  "C'est un point important que vous soulevez. Voici mon analyse..."
];

const getRandomResponse = () => {
  return botResponses[Math.floor(Math.random() * botResponses.length)];
};

const generateChatTitle = (firstMessage) => {
  const words = firstMessage.split(' ').slice(0, 6);
  return words.join(' ') + (firstMessage.split(' ').length > 6 ? '...' : '');
};

export const useChatGPT = () => {
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [isTyping, setIsTyping] = useState(false);

  const currentChat = chats.find(chat => chat.id === currentChatId);

  const createNewChat = useCallback(() => {
    const newChatId = Date.now().toString();
    const newChat = {
      id: newChatId,
      title: 'Nouvelle conversation',
      messages: [],
      timestamp: new Date()
    };
    
    setChats(prev => [newChat, ...prev]);
    setCurrentChatId(newChatId);
  }, []);

  const selectChat = useCallback((chatId) => {
    setCurrentChatId(chatId);
  }, []);

  const deleteChat = useCallback((chatId) => {
    setChats(prev => prev.filter(chat => chat.id !== chatId));
    if (currentChatId === chatId) {
      setCurrentChatId(null);
    }
  }, [currentChatId]);

  const renameChat = useCallback((chatId, newTitle) => {
    setChats(prev => prev.map(chat => 
      chat.id === chatId ? { ...chat, title: newTitle } : chat
    ));
  }, []);

  const sendMessage = useCallback((text) => {
    if (!currentChatId) {
      createNewChat();
      return;
    }

    const userMessage = {
      id: Date.now().toString(),
      text,
      isBot: false,
      timestamp: new Date(),
    };

    setChats(prev => prev.map(chat => {
      if (chat.id === currentChatId) {
        const updatedMessages = [...chat.messages, userMessage];
        const updatedTitle = chat.messages.length === 0 ? generateChatTitle(text) : chat.title;
        
        return {
          ...chat,
          messages: updatedMessages,
          title: updatedTitle,
          timestamp: new Date()
        };
      }
      return chat;
    }));

    setIsTyping(true);

    // Simulate bot response with realistic delay
    setTimeout(() => {
      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: getRandomResponse(),
        isBot: true,
        timestamp: new Date(),
      };

      setChats(prev => prev.map(chat => {
        if (chat.id === currentChatId) {
          return {
            ...chat,
            messages: [...chat.messages, botMessage],
            timestamp: new Date()
          };
        }
        return chat;
      }));

      setIsTyping(false);
    }, 2000 + Math.random() * 1500);
  }, [currentChatId, createNewChat]);

  return {
    chats,
    currentChat,
    currentChatId,
    isTyping,
    createNewChat,
    selectChat,
    deleteChat,
    renameChat,
    sendMessage,
  };
};