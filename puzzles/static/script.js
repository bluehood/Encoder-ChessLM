$(document).ready(function() {
    let board = null;
    let editorBoard = null; // Board editor instance
    let editorGame = new Chess(); // Chess.js instance for the editor
    let currentPuzzleFen = null;
    let puzzleGame = null; // Chess.js game instance
    let puzzleMoves = null; // Full sequence of puzzle moves
    let currentMoveIndex = 0; // Track which move in the sequence we're on
    let userTurn = false; // Track if it's user's turn to move - Start with false since computer moves first

    // Default starting position
    const startPosition = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

    // Helper function to generate FEN from board state
    function getFenFromPosition(position) {
        const tempGame = new Chess('8/8/8/8/8/8/8/8 w - - 0 1');
        tempGame.clear();
        for (const square in position) {
            const piece = position[square];
            const color = piece[0];
            const pieceType = piece[1].toLowerCase();
            tempGame.put({ type: pieceType, color: color }, square);
        }
        // Determine whose turn it is - This is tricky without full game context
        // Defaulting to white's turn for the editor. You might need more logic here.
        // Consider adding UI elements to set turn, castling rights, en passant.
        let fen = tempGame.fen().split(' ')[0]; // Get only the piece placement part
        fen += ' w - - 0 1'; // Assume white to move, no castling/en passant
        return fen;
    }

    // --- Board Editor Configuration ---
    function initializeBoardEditor() {
        const config = {
            draggable: true,
            position: startPosition,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
            dropOffBoard: 'trash', // Allow removing pieces
            sparePieces: true, // Show spare pieces for adding to the board
            showNotation: true,
            onChange: handleBoardEditorChange // Add the onChange handler
        };
        editorBoard = Chessboard('boardEditor', config);
        
        // Set the default FEN in the input
        $('#fenInput').val(startPosition);
        
        // Handle the "Load Position" button click
        $('#loadPositionBtn').click(function() {
            const fen = $('#fenInput').val().trim();
            if (validateFen(fen)) {
                editorBoard.position(fen); // onChange will trigger FEN update
                $('#inputError').hide();
            } else {
                $('#inputError').text('Invalid FEN position').show();
            }
        });
        
        // Handle the "Reset Board" button click
        $('#resetBoardBtn').click(function() {
            editorBoard.start(); // onChange will trigger FEN update to startPosition
            $('#inputError').hide();
        });
        
        // Ensure board resizes on window resize
        $(window).resize(function() {
            if (editorBoard) {
                editorBoard.resize();
            }
        });
    }

    // Function called when the board editor changes
    function handleBoardEditorChange(oldPos, newPos) {
        // Update the FEN input based on the new position
        try {
            const fen = getFenFromPosition(Chessboard.objToFen(newPos));
            $('#fenInput').val(fen);
            $('#inputError').hide();
        } catch (e) {
            console.error("Error updating FEN from editor change:", e);
        }
    }

    // Function to validate FEN (basic)
    function validateFen(fen) {
        try {
            new Chess(fen);
            return true;
        } catch (e) {
            return false;
        }
    }

    // --- Chessboard Configuration ---
    function initializeBoard(fen) {
        // Create a new chess.js instance for the puzzle
        puzzleGame = new Chess(fen);
        currentPuzzleFen = fen; // Store the FEN for the displayed puzzle
        
        // First, determine the player's color (opposite of who's to move in initial position)
        // In Lichess puzzles, if White is to move in the initial position, player will be Black and vice versa
        const playerColor = puzzleGame.turn() === 'w' ? 'black' : 'white';
        
        const config = {
            draggable: true,
            position: fen,
            onDrop: onDrop,
            orientation: playerColor, // Set orientation based on player's color
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
        };
        
        // Delay the board initialization slightly to ensure the containing div is visible and sized properly
        setTimeout(() => {
            board = Chessboard('puzzleBoard', config);
            
            // Handle window resize for the puzzle board
            $(window).resize(function() {
                if (board) {
                    board.resize();
                }
            });
            
            // After board is visible, trigger a resize to ensure it fits correctly
            board.resize();
            
            // Reset move tracking
            currentMoveIndex = 0;
            userTurn = false; // Start with computer's turn
            
            $('#moveResult').hide();
            
            // Make the computer's first move after a short delay
            setTimeout(function() {
                if (puzzleMoves && puzzleMoves.length > 0) {
                    makeComputerFirstMove();
                }
            }, 500);
        }, 100); // Slight delay to ensure DOM is ready
    }
    
    // Make the computer's first move when showing the puzzle
    function makeComputerFirstMove() {
        if (!puzzleMoves || puzzleMoves.length === 0 || !puzzleGame) {
            return false;
        }
        
        const firstMove = puzzleMoves[currentMoveIndex];
        
        // Extract source and target squares
        const source = firstMove.slice(0, 2);
        const target = firstMove.slice(2, 4);
        
        // Check for promotion
        let moveObj = {
            from: source,
            to: target
        };
        
        if (firstMove.length > 4) {
            moveObj.promotion = firstMove.slice(4);
        }
        
        // Make the move on the internal game
        try {
            puzzleGame.move(moveObj);
            
            // Update the board
            board.position(puzzleGame.fen());
            
            // Show a message indicating the computer's move
            $('#moveResult').text('Computer moved: ' + firstMove + '. Your turn!').removeClass('incorrect').addClass('correct').show();
            
            // Increment move index
            currentMoveIndex++;
            
            // Now it's user's turn
            userTurn = true;
            
            return true;
        } catch (e) {
            console.error("Error making computer's first move:", e);
            return false;
        }
    }
    
    // Make the opponent's move after player's correct move
    function makeOpponentMove() {
        if (!puzzleMoves || currentMoveIndex >= puzzleMoves.length || !puzzleGame) {
            return false;
        }
        
        userTurn = false;
        
        const nextMove = puzzleMoves[currentMoveIndex];
        
        // Convert UCI move to chess.js format if needed
        let moveObj = {};
        
        // Extract source and target squares
        const source = nextMove.slice(0, 2);
        const target = nextMove.slice(2, 4);
        
        // Check for promotion
        if (nextMove.length > 4) {
            moveObj = {
                from: source,
                to: target,
                promotion: nextMove.slice(4)
            };
        } else {
            moveObj = {
                from: source,
                to: target
            };
        }
        
        // Make the move on the internal game
        try {
            puzzleGame.move(moveObj);
            
            // Update the board
            board.position(puzzleGame.fen());
            
            // Increment the move index
            currentMoveIndex++;
            
            // It's user's turn again
            userTurn = true;
            
            return true;
        } catch (e) {
            console.error("Invalid opponent move:", nextMove, e);
            return false;
        }
    }
    
    // Check if puzzle is complete
    function isPuzzleComplete() {
        return currentMoveIndex >= puzzleMoves.length;
    }

    // --- Hint Functionality ---
    function provideHint() {
        if (!puzzleMoves || currentMoveIndex >= puzzleMoves.length || !puzzleGame || !userTurn) {
            return false;
        }
        
        // Get the expected move from the puzzle sequence
        const expectedMove = puzzleMoves[currentMoveIndex];
        
        // Extract source and target squares
        const source = expectedMove.slice(0, 2);
        const target = expectedMove.slice(2, 4);
        
        // Check for promotion
        let moveObj = {
            from: source,
            to: target
        };
        
        if (expectedMove.length > 4) {
            moveObj.promotion = expectedMove.slice(4);
        }
        
        // Make the move on the internal game
        try {
            puzzleGame.move(moveObj);
            
            // Update the board
            board.position(puzzleGame.fen());
            
            // Show hint message
            $('#moveResult').text('Hint applied: ' + expectedMove).removeClass('incorrect').addClass('correct').show();
            
            // Increment move index
            currentMoveIndex++;
            
            // Check if puzzle is complete
            if (isPuzzleComplete()) {
                $('#moveResult').text('Puzzle completed with hint!').removeClass('incorrect').addClass('correct').show();
                return true;
            }
            
            // Make the opponent's move after a short delay
            setTimeout(function() {
                const opponentMoved = makeOpponentMove();
                
                // If opponent move made and puzzle complete
                if (opponentMoved && isPuzzleComplete()) {
                    $('#moveResult').text('Puzzle completed!').removeClass('incorrect').addClass('correct').show();
                }
            }, 500);
            
            return true;
        } catch (e) {
            console.error("Error applying hint:", e);
            return false;
        }
    }

    // --- Bind hint button click handler ---
    $('#hintBtn').on('click', function() {
        provideHint();
    });

    // --- Move Handling ---
    function onDrop(source, target, piece, newPos, oldPos, orientation) {
        // Don't allow moves if it's not user's turn
        if (!userTurn) return 'snapback';
        
        // Check if this is a valid move in the puzzle's current state
        const move = {
            from: source,
            to: target
        };
        
        // Check for promotion
        const pieceType = piece.charAt(1).toLowerCase();
        const isPromotion = (pieceType === 'p') && 
                           ((target.charAt(1) === '8' && piece.charAt(0) === 'w') || 
                            (target.charAt(1) === '1' && piece.charAt(0) === 'b'));
        
        if (isPromotion) {
            move.promotion = 'q'; // Default to queen promotion for simplicity
        }
        
        // Get the expected move from the puzzle sequence
        const expectedMove = puzzleMoves[currentMoveIndex];
        
        // Convert move to UCI format for comparison
        let uciMove = source + target;
        if (move.promotion) {
            uciMove += move.promotion;
        }
        
        // Try to make the move on the internal chess.js instance
        try {
            const moveResult = puzzleGame.move(move);
            
            if (!moveResult) {
                return 'snapback'; // Invalid move according to chess rules
            }
            
            // Check if the move matches the expected move
            if (uciMove === expectedMove) {
                // Correct move!
                $('#moveResult').text('Correct!').removeClass('incorrect').addClass('correct').show();
                
                // Increment move index
                currentMoveIndex++;
                
                // Check if puzzle is complete
                if (isPuzzleComplete()) {
                    $('#moveResult').text('Puzzle completed successfully!').removeClass('incorrect').addClass('correct').show();
                    return; // Puzzle complete, no more moves needed
                }
                
                // Make the opponent's move after a short delay
                setTimeout(function() {
                    const opponentMoved = makeOpponentMove();
                    
                    // If opponent move made and puzzle complete
                    if (opponentMoved && isPuzzleComplete()) {
                        $('#moveResult').text('Puzzle completed successfully!').removeClass('incorrect').addClass('correct').show();
                    }
                }, 500);
                
                return; // Allow the move
            } else {
                // Incorrect move - revert the board
                $('#moveResult').html(`Incorrect. Try again!`)
                               .removeClass('correct').addClass('incorrect').show();
                
                // Undo the move in the internal game
                puzzleGame.undo();
                
                // Reset board to previous position
                board.position(puzzleGame.fen());
                
                return 'snapback';
            }
        } catch (e) {
            console.error("Error processing move:", e);
            return 'snapback';
        }
    }

    // --- API Interaction ---
    $('#findPuzzleBtn').on('click', function() {
        // The FEN input is now updated automatically by handleBoardEditorChange
        const inputFen = $('#fenInput').val().trim();
        
        $('#inputError').hide().text('');
        $('#serverError').hide().text('');
        $('#moveResult').hide().text('');
        $('.puzzle-section').hide();
        $('#loadingIndicator').show();

        if (!inputFen) {
            $('#inputError').text('Please enter a FEN string or set up a position.').show();
            $('#loadingIndicator').hide();
            return;
        }

        $.ajax({
            url: '/find_similar_puzzle',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ fen: inputFen }),
            success: function(data) {
                $('#loadingIndicator').hide();
                
                // Check for explicit error from backend
                if (data.error) {
                     $('#serverError').text(data.error).show(); // Display backend error
                     $('.puzzle-section').hide(); // Ensure puzzle section remains hidden
                     return;
                }
                
                // Proceed if no error
                $('.puzzle-section').show();

                $('#puzzleFen').text(data.puzzle_fen);
                $('#puzzleThemes').text(data.themes);

                // Store the full moves sequence
                puzzleMoves = data.moves;

                if (board) {
                    board.destroy(); // Clear previous board if any
                }
                
                // Since we're showing the puzzleSection now, we need to make
                // sure we initialize the board after it becomes visible
                initializeBoard(data.puzzle_fen); // Initialize the puzzle board
            },
            error: function(jqXHR, textStatus, errorThrown) {
                $('#loadingIndicator').hide();
                let errorMsg = 'An error occurred while fetching the puzzle.';
                if (jqXHR.responseJSON && jqXHR.responseJSON.detail) {
                    errorMsg = jqXHR.responseJSON.detail;
                } else if (jqXHR.statusText) {
                    errorMsg = `Error: ${jqXHR.statusText}`;
                }
                 // Display error specific to input or server
                if (jqXHR.status === 400 || jqXHR.status === 422) { // Bad Request or Unprocessable Entity (often used for validation errors)
                     $('#inputError').text(`Error: ${errorMsg}`).show();
                } else {
                     $('#serverError').text(errorMsg).show();
                }
                $('.puzzle-section').hide(); // Hide puzzle section on error
            }
        });
    });
    
    // Initialize the board editor when the page loads
    initializeBoardEditor();
}); 