const controls = document.querySelector('.work-controls');
const buttons = [...document.querySelectorAll('[data-filter]')];
const works = [...document.querySelectorAll('.bibitem')];
const count = document.querySelector('#work-count');
const track = document.querySelector('.selection-track');

// Each tick corresponds to a real publication and keeps its place in the index.
const ticks = works.map(() => {
  const tick = document.createElement('i');
  tick.className = 'selected';
  track.append(tick);
  return tick;
});

buttons.forEach(button => button.addEventListener('click', () => {
  const filter = button.dataset.filter;
  buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  let visible = 0;
  works.forEach((work, index) => {
    const selected = filter === 'all' || work.dataset.category === filter;
    work.hidden = !selected;
    ticks[index].classList.toggle('selected', selected);
    if (selected) visible++;
  });
  count.textContent = `${String(visible).padStart(2, '0')} / ${String(works.length).padStart(2, '0')} works`;
}));
controls.hidden = false;
